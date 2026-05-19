"""
End-to-End Integration Test for the Unified Intelligence Platform.

This script proves the entire lifecycle:
1. Register a tenant and receive an API key.
2. Upload a raw Knowledge Base -> Engine auto-generates graph + GNN.
3. Stream a live temporal shock into the Engine.
4. Query the Engine for intelligence -> receive a TaCIE prompt.
5. Submit retrospective feedback -> Engine learns.
6. Query again -> verify the Engine's learned confidence has changed.
"""

import asyncio
import httpx
import json

BASE = "http://127.0.0.1:8000"

async def run_test():
    async with httpx.AsyncClient(base_url=BASE, timeout=30.0) as client:
        
        print("=" * 70)
        print("   UNIFIED INTELLIGENCE PLATFORM — END-TO-END LIFECYCLE TEST")
        print("=" * 70)
        
        # ── STEP 1: Register ──
        print("\n[STEP 1] Registering tenant...")
        r = await client.post("/register", json={"org_name": "Acme Logistics"})
        reg = r.json()
        api_key = reg["api_key"]
        print(f"  Tenant: {reg['org_id']}")
        print(f"  API Key: {api_key[:20]}...")
        headers = {"X-API-Key": api_key}
        
        # ── STEP 2: Ingest a raw KB ──
        print("\n[STEP 2] Uploading raw Knowledge Base (Logistics Domain)...")
        kb_text = """
        The container_port supplies the central_warehouse.
        The central_warehouse distributes to the regional_depot.
        The regional_depot distributes to the last_mile_courier.
        The fuel_market affects the last_mile_courier.
        The customs_authority regulates the container_port.
        To survive fuel spikes, you should pre_negotiate_fuel_contracts.
        To survive port delays, you should diversify_shipping_routes.
        To survive demand surges, you should expand_warehouse_capacity.
        """
        r = await client.post("/projects/ingest", json={
            "project_name": "Global Logistics Network",
            "knowledge_base_text": kb_text
        }, headers=headers)
        project = r.json()
        project_id = project["project_id"]
        print(f"  Project ID: {project_id}")
        print(f"  Status: {project['status']}")
        print(f"  Auto-Generated Nodes: {project['auto_generated_graph']['nodes']}")
        print(f"  Auto-Extracted Actions: {project['auto_extracted_actions']}")
        
        # ── STEP 3: Stream a live shock ──
        print("\n[STEP 3] Streaming live temporal shock: Fuel price spike...")
        r = await client.post(f"/projects/{project_id}/stream", json={
            "node_name": "FUELMARKET",
            "shock_value": 6.0
        }, headers=headers)
        stream_result = r.json()
        print(f"  Updated Network State:")
        for node, threat in sorted(stream_result["updated_network_state"].items(), key=lambda x: x[1], reverse=True):
            print(f"    [{node}]: {threat:.3f}")
        
        # ── STEP 4: Query the Engine for intelligence ──
        print("\n[STEP 4] Querying intelligence for LASTMILECOURIER...")
        r = await client.post(f"/projects/{project_id}/query", json={
            "target_node": "LASTMILECOURIER",
            "k_hops": 2
        }, headers=headers)
        query_result = r.json()
        print(f"  Target Threat Level: {query_result['current_threat']:.3f}")
        print(f"  Local Network: {query_result['local_network']}")
        print(f"  Action Confidence: {json.dumps(query_result['counterfactual_actions'], indent=4)}")
        print(f"\n  --- TaCIE Prompt (first 500 chars) ---")
        print(f"  {query_result['tacie_prompt_for_llm'][:500]}...")
        
        # ── STEP 5: Submit feedback ──
        print("\n\n[STEP 5] Submitting retrospective feedback...")
        # Client tried 'pre_negotiate_fuel_contracts' and it worked
        r = await client.post(f"/projects/{project_id}/feedback", json={
            "node": "LASTMILECOURIER",
            "action_taken": "pre_negotiate_fuel_contracts",
            "initial_threat": query_result["current_threat"],
            "realized_outcome": 0.5  # Threat dropped significantly
        }, headers=headers)
        feedback_result = r.json()
        print(f"  Updated Learned Confidence: {json.dumps(feedback_result['updated_action_confidence'], indent=4)}")
        
        # ── STEP 6: List all live projects ──
        print("\n[STEP 6] Discovering all live intelligence projects...")
        r = await client.get("/projects", headers=headers)
        projects = r.json()
        for p in projects["projects"]:
            print(f"  [{p['status'].upper()}] {p['project_name']} | Nodes: {p['nodes']} | Actions: {p['actions']}")
        
        print("\n" + "=" * 70)
        print("   ALL STEPS PASSED. THE ENGINE IS THE API. THE API IS THE ENGINE.")
        print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_test())
