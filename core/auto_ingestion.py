import networkx as nx
import re
import json

class UniversalKnowledgeGraphCompiler:
    def __init__(self):
        """
        Stage 1: The Auto-Ingestion NLP Engine.
        This class is domain-agnostic. It reads raw text and automatically generates
        a Labeled Property Graph (LPG) representing the underlying architecture of the domain.
        """
        self.graph = nx.DiGraph()
        self.extracted_actions = set()

    async def parse_knowledge_base(self, raw_text: str, llm_client=None):
        """
        Ingests a generic knowledge base (text) and runs the extraction pipeline.
        If an llm_client is provided, it tries to do a structured extraction via LLM first.
        Otherwise, it falls back to the local regex heuristics.
        """
        print("[NLP] Starting Universal Auto-Ingestion Pipeline...")
        
        # 1. Attempt LLM-assisted structured extraction first
        if llm_client is not None:
            success = await self._parse_with_llm(raw_text, llm_client)
            if success:
                print(f"[Graph Compiler] Successfully auto-generated graph schema (LLM): {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges.")
                print(f"[Action Space] Auto-detected possible mitigations (LLM): {len(self.extracted_actions)} actions.")
                return self.graph
        
        # 2. Heuristic Regex fallback
        sentences = raw_text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            self._extract_triplets(sentence)
            self._extract_action_space(sentence)

        print(f"[Graph Compiler] Successfully auto-generated graph schema (Regex): {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges.")
        print(f"[Action Space] Auto-detected possible mitigations (Regex): {len(self.extracted_actions)} actions.")
        return self.graph

    async def _parse_with_llm(self, raw_text: str, llm_client) -> bool:
        """
        Queries the LLM with a structured schema prompt to build the graph.
        """
        prompt = f"""
Analyze the following unstructured knowledge base and extract the network graph structure and mitigations.
Return ONLY a valid JSON object with the following structure:
{{
  "triplets": [
    {{"source": "NODE_A", "relationship": "supplies", "target": "NODE_B", "weight": 1.0}}
  ],
  "actions": [
    "action_strategy_1"
  ]
}}

Ensure that relationship matches one of: 'supplies', 'depends_on', 'regulates', 'affects', 'distributes_to'.
Ensure node names are upper case alphanumeric with underscores.

Knowledge Base:
\"\"\"
{raw_text}
\"\"\"
"""
        try:
            res = await llm_client.generate(prompt, target_node="ingestion", threat_level=0.0, actions={})
            advisory = res.get("advisory", "")
            # Find JSON boundaries
            start = advisory.find("{")
            end = advisory.rfind("}") + 1
            if start != -1 and end != -1:
                data = json.loads(advisory[start:end])
                for t in data.get("triplets", []):
                    u = self._clean_entity(t["source"])
                    v = self._clean_entity(t["target"])
                    if u and v:
                        if not self.graph.has_node(u):
                            self.graph.add_node(u, dynamic_stress=0.0)
                        if not self.graph.has_node(v):
                            self.graph.add_node(v, dynamic_stress=0.0)
                        self.graph.add_edge(u, v, relation_type=t.get("relationship", "affects"), weight=float(t.get("weight", 1.0)))
                for a in data.get("actions", []):
                    self.extracted_actions.add(a.strip())
                return True
        except Exception as e:
            print(f"[NLP Info] LLM Ingestion failed: {e}. Falling back to heuristics.")
        return False

    def _extract_triplets(self, sentence: str):
        """
        Extracts Subject-Predicate-Object triplets to form the graph nodes and edges.
        """
        # A simple domain-agnostic heuristic parser for the prototype
        # Looking for relationship keywords: 'supplies', 'depends on', 'regulates', 'affects'
        relation_keywords = ['supplies', 'depends on', 'regulates', 'affects', 'impacts', 'distributes to']
        
        for rel in relation_keywords:
            if rel in sentence.lower():
                parts = re.split(rf'(?i)\s+{rel}\s+', sentence)
                if len(parts) == 2:
                    # Clean the extracted node names
                    subject_node = self._clean_entity(parts[0])
                    object_node = self._clean_entity(parts[1])
                    
                    if subject_node and object_node:
                        # Auto-create nodes
                        if not self.graph.has_node(subject_node):
                            self.graph.add_node(subject_node, dynamic_stress=0.0)
                        if not self.graph.has_node(object_node):
                            self.graph.add_node(object_node, dynamic_stress=0.0)
                            
                        # Auto-create edge
                        weight = 1.0 # Default starting weight
                        self.graph.add_edge(subject_node, object_node, relation_type=rel, weight=weight)

    def _extract_action_space(self, sentence: str):
        """
        (Missing Element B): Automatically extracts the "Action Space" (mitigations/recommendations)
        from the Knowledge Base so the engine knows what actions it can recommend.
        """
        action_keywords = ['mitigate by', 'should', 'can hedge via', 'action:', 'recommendation:']
        for keyword in action_keywords:
            if keyword in sentence.lower():
                idx = sentence.lower().find(keyword)
                action = sentence[idx + len(keyword):].strip()
                if action:
                    self.extracted_actions.add(action)

    def _clean_entity(self, text: str) -> str:
        # Remove common stop words and punctuation for cleaner node names
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = text.split()
        stop_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those'}
        cleaned = [w for w in words if w.lower() not in stop_words]
        return "_".join(cleaned).upper()

    def get_graph_schema_json(self):
        """
        Exports the dynamically generated schema so the GNN can auto-configure itself.
        """
        nodes = list(self.graph.nodes(data=True))
        edges = list(self.graph.edges(data=True))
        return json.dumps({"nodes": nodes, "edges": edges}, indent=2)

if __name__ == "__main__":
    # Test the Auto-Ingestion with a generic Supply Chain Knowledge Base (NOT Nigerian Economics)
    mock_kb = """
    The microchip_factory supplies the automotive_plant. 
    The lithium_mine supplies the battery_factory.
    The battery_factory supplies the automotive_plant.
    The global_shipping_lane affects the lithium_mine.
    The central_bank regulates the microchip_factory.
    To survive shipping delays, you should increase_local_inventory.
    To survive inflation, you can hedge via forward_contracts.
    """
    
    import asyncio
    compiler = UniversalKnowledgeGraphCompiler()
    graph = asyncio.run(compiler.parse_knowledge_base(mock_kb))
    
    print("\n--- Dynamically Generated Universal Schema ---")
    print(compiler.get_graph_schema_json())
    print("\n--- Auto-Extracted Action Space ---")
    for a in compiler.extracted_actions:
        print(f"- {a}")
