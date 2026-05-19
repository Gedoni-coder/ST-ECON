import asyncio
import httpx
import sys
import time
import subprocess
import os

async def run_test():
    db_file = os.path.join("data", "platform.db")
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
            print("[Clean] Cleared previous database file.")
        except Exception as e:
            print(f"[Warn] Could not delete old database file: {e}")

    # 1. Start Server in background on port 8007
    print("[Test] Launching Server on port 8007...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8007"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to bind
    time.sleep(3)
    
    api_key = None
    project_id = None
    
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8007", timeout=15) as client:
            # 2. Register Tenant
            print("[Test] Registering tenant...")
            resp = await client.post("/register", json={"org_name": "Persistence Corp"})
            reg_data = resp.json()
            api_key = reg_data["api_key"]
            headers = {"x-api-key": api_key}
            print(f"[Test] Registered successfully. Key: {api_key}")
            
            # 3. Ingest Project
            kb_text = """
            CHIPCOMPANY supplies DISTRIBUTIONHUB.
            DISTRIBUTIONHUB distributes to RETAILSTORE.
            recommendation: increase_retail_inventory.
            recommendation: source_alternative_chips.
            """
            print("[Test] Ingesting knowledge base...")
            resp = await client.post(
                "/projects/ingest", 
                json={"project_name": "Hardware Pipeline", "knowledge_base_text": kb_text},
                headers=headers
            )
            print(f"[Test] Ingest response status: {resp.status_code}")
            print(f"[Test] Ingest response JSON: {resp.text}")
            ingest_data = resp.json()
            project_id = ingest_data["project_id"]
            print(f"[Test] Ingested project ID: {project_id}")
            
            # Inject a shock to mutate GNN state before termination
            print("[Test] Streaming shock value...")
            await client.post(
                f"/projects/{project_id}/stream",
                json={"node_name": "CHIPCOMPANY", "shock_value": 3.5},
                headers=headers
            )
            
            # Query status to verify state before shutdown
            print("[Test] Querying state before shutdown...")
            resp = await client.post(
                f"/projects/{project_id}/query",
                json={"target_node": "RETAILSTORE", "k_hops": 2},
                headers=headers
            )
            state_before = resp.json()
            print(f" -> Initial Threat score at RETAILSTORE: {state_before['current_threat']:.3f}")
            
    finally:
        # 4. Terminate Server (Kill Process)
        print("[Test] Terminating server to clear memory...")
        server_process.terminate()
        server_process.wait()
        print("[Test] Server stopped. Cache cleared.")

    # 5. Restart Server in background
    print("[Test] Restarting Server on port 8007 (Cold Boot)...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8007"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(3)
    
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8007", timeout=15) as client:
            headers = {"x-api-key": api_key}
            
            # 6. Retrieve Project WITHOUT Ingesting it again
            print("[Test] Querying project status from persistent store...")
            resp = await client.post(
                f"/projects/{project_id}/query",
                json={"target_node": "RETAILSTORE", "k_hops": 2},
                headers=headers
            )
            
            if resp.status_code == 200:
                state_after = resp.json()
                print("\n============================================================")
                print("SUCCESS: State successfully reloaded from SQLite database!")
                print(f"Nodes loaded in threat map: {list(state_after['threat_map'].keys())}")
                print(f"Advisory backend: {state_after['advisory_backend']}")
                print("============================================================\n")
            else:
                print(f"Error querying project after restart: {resp.status_code} | {resp.text}")
                
    finally:
        print("[Test] Shutting down restarted server...")
        server_process.terminate()
        server_process.wait()
        print("[Test] Done.")

if __name__ == "__main__":
    asyncio.run(run_test())
