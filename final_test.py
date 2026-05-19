import asyncio
import httpx
import json

async def final_test():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8012", timeout=30) as c:
        # Register
        r = await c.post("/register", json={"org_name": "GlobalHealth Corp"})
        key = r.json()["api_key"]
        h = {"X-API-Key": key}

        # Ingest a healthcare KB
        kb = """
        The pharma_supplier supplies the national_depot.
        The national_depot distributes to the regional_hospital.
        The regional_hospital distributes to the local_clinic.
        The fuel_market affects the pharma_supplier.
        The health_ministry regulates the national_depot.
        To survive supply shortages, you should stockpile_critical_drugs.
        To survive regulatory audits, you should enforce_compliance_logs.
        """
        r = await c.post("/projects/ingest", json={"project_name": "Healthcare Supply Chain", "knowledge_base_text": kb}, headers=h)
        data = r.json()
        pid = data["project_id"]
        nodes = data["auto_generated_graph"]["nodes"]
        actions = data["auto_extracted_actions"]

        # Stream a live fuel crisis shock
        await c.post(f"/projects/{pid}/stream", json={"node_name": "FUELMARKET", "shock_value": 9.0}, headers=h)

        # Query intelligence
        r = await c.post(f"/projects/{pid}/query", json={"target_node": "REGIONALHOSPITAL"}, headers=h)
        q = r.json()

        print("=" * 60)
        print("  COMPLETE PLATFORM TEST -- ALL LAYERS")
        print("=" * 60)
        print(f"Nodes auto-generated: {nodes}")
        print(f"Actions auto-extracted: {actions}")
        threat = q["current_threat"]
        print(f"Threat at REGIONALHOSPITAL: {threat:.3f}")
        print(f"Network: {q['local_network']}")
        print(f"Advisory Backend: {q['advisory_backend']} | Model: {q['advisory_model']}")
        print()
        print("--- ADVISORY ---")
        print(q["advisory"])
        print()
        print("--- TRANSFUSION ACCELERATION ---")
        for k, v in q["transfusion_acceleration"].items():
            print(f"  {k}: {v}")
        print("=" * 60)
        print("ALL COMPLETE.")

asyncio.run(final_test())
