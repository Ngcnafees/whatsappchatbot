import httpx
from salesforce_auth import get_sf_token

async def sf_product_search(code: str):
    token, instance_url = await get_sf_token()
    url = f"{instance_url}/services/apexrest/product/search"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, headers=headers, params={"code": code})

        # If token expired unexpectedly, refresh once and retry
        if resp.status_code == 401:
            token, instance_url = await get_sf_token()
            url = f"{instance_url}/services/apexrest/product/search"
            headers = {"Authorization": f"Bearer {token}"}
            resp = await client.get(url, headers=headers, params={"code": code})

        resp.raise_for_status()
        return resp.json()
