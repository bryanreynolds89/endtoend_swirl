import openai
import re
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Document,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)

import psycopg2
from psycopg2.extras import RealDictCursor


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding


### Item Description Retrieval Tool


@traceable(name="retrieve_data", run_type="retriever")
def retrieve_items_data(query, k=5):
    query_embedding = get_embedding(query)

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(query=query_embedding, using="text-embedding-3-small", limit=20),
            Prefetch(query=Document(text=query, model="qdrant/bm25"), using="bm25", limit=20),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload.get("parent_asin"))
        retrieved_context.append(result.payload.get("description"))
        retrieved_context_ratings.append(result.payload.get("average_rating"))
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }


@traceable(name="format_retrieved_context", run_type="prompt")
def process_items_context(context):
    formatted_context = ""

    for id_val, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"- ID: {id_val}, rating: {rating}, description: {chunk}\n"

    return formatted_context


def get_formatted_items_context(query: str, top_k: int = 5) -> str:
    """Get the top k context chunks for a given query, each representing an inventory item.

    Args:
        query: The query to retrieve relevant inventory context for.
        top_k: Number of items to retrieve.

    Returns:
        A single formatted string containing the retrieved items, including IDs and ratings.
    """
    context = retrieve_items_data(query, top_k)
    return process_items_context(context)


### Item Reviews Retrieval Tool


@traceable(name="retrieve_reviews_data", run_type="retriever")
def retrieve_reviews_data(query, item_list, k=5):
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-reviews",
        prefetch=[
            Prefetch(
                query=query_embedding,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(any=item_list),
                        )
                    ]
                ),
                limit=20,
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.payload.get("parent_asin"))
        retrieved_context.append(result.payload.get("text"))
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }


@traceable(name="format_retrieved_reviews_context", run_type="prompt")
def process_reviews_context(context):
    formatted_context = ""

    for id_val, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- ID: {id_val}, review: {chunk}\n"

    return formatted_context


def get_formatted_reviews_context(query: str, item_list: list, top_k: int = 15) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.

    Args:
        query: The query to retrieve relevant reviews for.
        item_list: List of item IDs to filter reviews by.
        top_k: Number of reviews to retrieve.

    Returns:
        A single formatted string containing the retrieved reviews, including IDs.
    """
    context = retrieve_reviews_data(query, item_list, top_k)
    return process_reviews_context(context)


### Shopping Cart Tools


def _db_conn():
    """
    Docker compose friendly connection.
    Other tools should use this instead of localhost.
    """
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password",
    )
    conn.autocommit = True
    return conn


def _fetch_item_payload_by_parent_asin(qdrant_client: QdrantClient, parent_asin: str) -> dict | None:
    points, _ = qdrant_client.scroll(
        collection_name="Amazon-items-collection-01-hybrid-search",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="parent_asin",
                    match=MatchValue(value=parent_asin),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    return points[0].payload or {}


def _extract_image_url(payload: dict) -> str | None:
    for k in ["image", "image_url", "main_image", "product_image_url"]:
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    imgs = payload.get("images")
    if isinstance(imgs, list) and imgs:
        first = imgs[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
        if isinstance(first, dict):
            for kk in ["url", "link"]:
                vv = first.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()

    return None


def _extract_title(payload: dict) -> str | None:
    for k in ["title", "product_title", "name"]:
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_description(payload: dict) -> str | None:
    v = payload.get("description")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _extract_price(payload: dict) -> float | None:
    """
    Attempts to normalize a price from a variety of common payload formats.
    Returns None when no usable numeric price is present.
    """
    def _to_float(val) -> float | None:
        if val is None:
            return None

        if isinstance(val, (int, float)):
            return float(val)

        if isinstance(val, str):
            s = val.strip()
            if not s:
                return None

            lowered = s.lower()
            if lowered in {"n/a", "na", "none", "null", "free", "see price", "see price in cart"}:
                return None

            # Handle range-like strings such as "$12.99 - $19.99" or "12.99 to 19.99"
            if ("-" in s) or (" to " in lowered):
                parts = re.split(r"\s*(?:-|to)\s*", s, flags=re.IGNORECASE)
                for p in parts:
                    candidate = _to_float(p)
                    if candidate is not None:
                        return candidate

            # Extract first numeric token if the string contains extra text
            m = re.search(r"(\d+(?:\.\d+)?)", s.replace(",", ""))
            if not m:
                return None

            num = m.group(1)
            try:
                return float(num)
            except ValueError:
                return None

        if isinstance(val, dict):
            for k in ["value", "amount", "price", "usd", "raw"]:
                if k in val:
                    parsed = _to_float(val.get(k))
                    if parsed is not None:
                        return parsed

            for k in ["$numberDouble", "$numberInt", "$numberLong", "$numberDecimal"]:
                if k in val:
                    parsed = _to_float(val.get(k))
                    if parsed is not None:
                        return parsed
            return None

        if isinstance(val, list):
            for item in val:
                parsed = _to_float(item)
                if parsed is not None:
                    return parsed
            return None

        return None

    for key in ["price", "price_usd", "discounted_price", "list_price"]:
        if key in payload:
            parsed = _to_float(payload.get(key))
            if parsed is not None:
                return parsed

    return None


def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:
    """
    Retrieve all items in a user's shopping cart.
    Always returns hydrated fields when possible, even if DB rows have nulls.
    """

    conn = _db_conn()
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
            SELECT
                product_id,
                price,
                quantity,
                currency,
                product_image_url,
                (price * quantity) as total_price
            FROM shopping_carts.shopping_cart_items
            WHERE user_id = %s AND shopping_cart_id = %s
            ORDER BY added_at DESC
        """
        cursor.execute(query, (user_id, cart_id))
        rows = [dict(r) for r in cursor.fetchall()]

    hydrated = []
    for row in rows:
        product_id = row.get("product_id")

        payload = _fetch_item_payload_by_parent_asin(qdrant_client, product_id) if product_id else None
        payload = payload or {}

        row_price = row.get("price")
        extracted_price = _extract_price(payload)

        # Prefer DB price if present; otherwise fall back to extracted price.
        price = row_price if row_price is not None else extracted_price

        img = row.get("product_image_url")
        if not img:
            img = _extract_image_url(payload)

        currency = row.get("currency") or payload.get("currency") or "USD"
        quantity = row.get("quantity") or 0

        total_price = None
        if isinstance(price, (int, float)) and isinstance(quantity, int):
            total_price = float(price) * quantity

        # Flag when price was effectively missing in the source payload.
        # This helps debugging and UI messaging without breaking the list return shape.
        price_missing = False
        if (row_price is None or float(row_price) == 0.0) and extracted_price is None:
            raw_payload_price = payload.get("price")
            if raw_payload_price is None:
                price_missing = True

        hydrated.append(
            {
                "product_id": product_id,
                "title": _extract_title(payload),
                "description": _extract_description(payload),
                "price": float(price) if isinstance(price, (int, float)) else None,
                "quantity": quantity,
                "currency": currency,
                "product_image_url": img,
                "total_price": total_price,
                "price_missing": price_missing,
            }
        )

    return hydrated


def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> list[dict]:
    """
    Add items, then return the full updated cart as structured data.

    Key behavior change:
    - Tool exceptions are avoided for missing or unparsable price.
    - When a price is missing, the item is still inserted with price=0.0 so downstream
      tool call plumbing remains valid and the UI remains deterministic.
    - The cart hydration includes a boolean field price_missing to signal the condition.
    """

    conn = _db_conn()
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        for item in items:
            product_id = item.get("product_id")
            quantity = item.get("quantity")

            # Input validation without raising tool-breaking exceptions.
            if not product_id:
                continue
            if quantity is None:
                continue
            if not isinstance(quantity, int) or quantity <= 0:
                continue

            payload = _fetch_item_payload_by_parent_asin(qdrant_client, product_id)
            if payload is None:
                continue

            product_image_url = _extract_image_url(payload)
            price = _extract_price(payload)
            currency = payload.get("currency") or "USD"

            # If price is missing, store 0.0 rather than raising.
            # This avoids breaking the tool call sequence and keeps the cart view stable.
            if price is None:
                price = 0.0

            check_query = """
                SELECT id, quantity
                FROM shopping_carts.shopping_cart_items
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone()

            if existing_item:
                new_quantity = existing_item["quantity"] + quantity
                update_query = """
                    UPDATE shopping_carts.shopping_cart_items
                    SET
                        quantity = %s,
                        price = %s,
                        currency = %s,
                        product_image_url = COALESCE(%s, product_image_url)
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                """
                cursor.execute(
                    update_query,
                    (new_quantity, price, currency, product_image_url, user_id, cart_id, product_id),
                )
            else:
                insert_query = """
                    INSERT INTO shopping_carts.shopping_cart_items (
                        user_id,
                        shopping_cart_id,
                        product_id,
                        price,
                        quantity,
                        currency,
                        product_image_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(
                    insert_query,
                    (user_id, cart_id, product_id, price, quantity, currency, product_image_url),
                )

    return get_shopping_cart(user_id=user_id, cart_id=cart_id)


def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> list[dict]:
    """
    Remove an item, then return the updated cart.
    """

    conn = _db_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
            DELETE FROM shopping_carts.shopping_cart_items
            WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
        """
        cursor.execute(query, (user_id, cart_id, product_id))

    return get_shopping_cart(user_id=user_id, cart_id=cart_id)


### Warehouse Manager Agent Tools


def check_warehouse_availability(items: list[dict]) -> dict:
    """Check availability of items across warehouses, including partial fulfillment options.

    Args:
        items: A list of items to check. Each item is a dictionary with keys: product_id, quantity.

    Returns:
        A dictionary containing:
        - can_fulfill_completely: bool indicating if all items can be fulfilled from at least one warehouse
        - warehouses_full_fulfillment: list of warehouses that can fulfill the entire order
        - warehouses_partial_fulfillment: list of warehouses with partial availability
        - unavailable_items: list of items that cannot be fulfilled from any warehouse
        - details: detailed breakdown per warehouse with availability for each item
    """

    conn = _db_conn()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "can_fulfill_completely": False,
                "warehouses_full_fulfillment": [],
                "warehouses_partial_fulfillment": [],
                "unavailable_items": [],
                "details": [],
            }

            warehouse_query = """
                SELECT DISTINCT warehouse_id, warehouse_name, warehouse_location
                FROM warehouses.inventory
            """
            cursor.execute(warehouse_query)
            warehouses = cursor.fetchall()

            for warehouse in warehouses:
                warehouse_can_fulfill_all = True
                has_any_availability = False
                warehouse_details = {
                    "warehouse_id": warehouse["warehouse_id"],
                    "warehouse_name": warehouse["warehouse_name"],
                    "warehouse_location": warehouse["warehouse_location"],
                    "items": [],
                    "can_fulfill_all": False,
                    "has_partial": False,
                }

                for item in items:
                    product_id = item.get("product_id")
                    requested_quantity = item.get("quantity")

                    if not product_id or requested_quantity is None:
                        continue
                    if not isinstance(requested_quantity, int) or requested_quantity <= 0:
                        continue

                    availability_query = """
                        SELECT
                            product_id,
                            total_quantity,
                            reserved_quantity,
                            (total_quantity - reserved_quantity) AS available_quantity
                        FROM warehouses.inventory
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(availability_query, (warehouse["warehouse_id"], product_id))
                    inventory = cursor.fetchone()

                    available_qty = int(inventory["available_quantity"]) if inventory else 0

                    item_detail = {
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "available": available_qty,
                        "can_fulfill_completely": available_qty >= requested_quantity,
                        "can_fulfill_partially": 0 < available_qty < requested_quantity,
                    }

                    warehouse_details["items"].append(item_detail)

                    if available_qty < requested_quantity:
                        warehouse_can_fulfill_all = False

                    if available_qty > 0:
                        has_any_availability = True

                if warehouse_can_fulfill_all and warehouse_details["items"]:
                    warehouse_details["can_fulfill_all"] = True
                    result["warehouses_full_fulfillment"].append(
                        {
                            "warehouse_id": warehouse["warehouse_id"],
                            "warehouse_name": warehouse["warehouse_name"],
                            "warehouse_location": warehouse["warehouse_location"],
                        }
                    )
                elif has_any_availability and warehouse_details["items"]:
                    warehouse_details["has_partial"] = True
                    result["warehouses_partial_fulfillment"].append(
                        {
                            "warehouse_id": warehouse["warehouse_id"],
                            "warehouse_name": warehouse["warehouse_name"],
                            "warehouse_location": warehouse["warehouse_location"],
                        }
                    )

                result["details"].append(warehouse_details)

            for item in items:
                product_id = item.get("product_id")
                requested_quantity = item.get("quantity")

                if not product_id or requested_quantity is None:
                    continue
                if not isinstance(requested_quantity, int) or requested_quantity <= 0:
                    continue

                total_available_query = """
                    SELECT
                        product_id,
                        SUM(total_quantity - reserved_quantity) AS total_available
                    FROM warehouses.inventory
                    WHERE product_id = %s
                    GROUP BY product_id
                """
                cursor.execute(total_available_query, (product_id,))
                total_available = cursor.fetchone()

                total_available_qty = int(total_available["total_available"]) if total_available else 0

                if total_available_qty < requested_quantity:
                    result["unavailable_items"].append(
                        {
                            "product_id": product_id,
                            "requested": requested_quantity,
                            "total_available_across_warehouses": total_available_qty,
                            "shortage": requested_quantity - total_available_qty,
                        }
                    )

            result["can_fulfill_completely"] = (
                len(result["warehouses_full_fulfillment"]) > 0 and len(result["unavailable_items"]) == 0
            )

            return result

    finally:
        conn.close()


def reserve_warehouse_items(reservations: list[dict]) -> dict:
    """Reserve items from multiple warehouses in a single transaction.

    Args:
        reservations: A list of reservations. Each reservation is a dictionary with keys:
                     - warehouse_id: The warehouse to reserve from
                     - product_id: The product to reserve
                     - quantity: The quantity to reserve

    Returns:
        A dictionary containing:
        - success: bool indicating if all reservations were successful
        - reserved_items: list of successfully reserved items
        - failed_items: list of items that could not be reserved
    """

    conn = _db_conn()
    conn.autocommit = False

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "success": False,
                "reserved_items": [],
                "failed_items": [],
            }

            for reservation in reservations:
                warehouse_id = reservation.get("warehouse_id")
                product_id = reservation.get("product_id")
                quantity = reservation.get("quantity")

                if not warehouse_id or not product_id or quantity is None:
                    result["failed_items"].append(
                        {
                            "product_id": product_id,
                            "warehouse_id": warehouse_id,
                            "requested": quantity,
                            "available": 0,
                            "reason": "invalid_input",
                        }
                    )
                    continue

                if not isinstance(quantity, int) or quantity <= 0:
                    result["failed_items"].append(
                        {
                            "product_id": product_id,
                            "warehouse_id": warehouse_id,
                            "requested": quantity,
                            "available": 0,
                            "reason": "invalid_quantity",
                        }
                    )
                    continue

                check_query = """
                    SELECT
                        warehouse_id,
                        product_id,
                        warehouse_name,
                        warehouse_location,
                        total_quantity,
                        reserved_quantity,
                        (total_quantity - reserved_quantity) AS available_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id = %s AND product_id = %s
                    FOR UPDATE
                """
                cursor.execute(check_query, (warehouse_id, product_id))
                inventory = cursor.fetchone()

                available_qty = int(inventory["available_quantity"]) if inventory else 0

                if inventory and available_qty >= quantity:
                    update_query = """
                        UPDATE warehouses.inventory
                        SET reserved_quantity = reserved_quantity + %s
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(update_query, (quantity, warehouse_id, product_id))

                    result["reserved_items"].append(
                        {
                            "product_id": product_id,
                            "quantity": quantity,
                            "warehouse_id": warehouse_id,
                            "warehouse_name": inventory["warehouse_name"],
                            "warehouse_location": inventory["warehouse_location"],
                        }
                    )
                else:
                    result["failed_items"].append(
                        {
                            "product_id": product_id,
                            "warehouse_id": warehouse_id,
                            "requested": quantity,
                            "available": available_qty,
                            "reason": "insufficient_stock" if inventory else "not_in_warehouse",
                        }
                    )

            if len(result["failed_items"]) == 0:
                conn.commit()
                result["success"] = True
            else:
                conn.rollback()
                result["success"] = False

            return result

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()