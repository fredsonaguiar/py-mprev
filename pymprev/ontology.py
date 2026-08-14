
class Ontology:
    def __init__(self):
        # Nodes
        self.nodes_schema = {
            # AI/ML Aspects
            # "Aiml_system": {
            #     "properties": ["name", "description"],
            #     "description": "systems and subsystems, models, tools and products based on or using AI/ML",
            # },
            "Aiml_element": {
                "properties": ["definition", "purpose", "name", "description"],
                "description": "elements or components. systems and subsystems, models, tools and products based on or using AI/ML: Neural Nets, Models, etc",
            },
            "Aiml_resource": {
                "properties": ["definition", "name", "description"],
                "description": "training sets, generators, accelerators (GPU/FPGA), libraries, parameters, hyperparameters",
            },
            "Aiml_lifecycle_aspect": {
                "properties": ["key_deliverable", "definition", "name", "description"],
                "description": "Aspects related to AI/ML development and maintenance: implementation, training, validation, deployment, etc",
            },

            # 
            "Risk_factor": {
                "properties": ["name", "description"],
                "description": "Risks system or component may be subject to: failures, biases, dataset distribution shifts, operational hazards, among others",
            },
            # "Risk_requirement": {
            #     "properties": ["name", "description"],
            #     "description": "Requirements a AI/ML system must comply with: functional, quantitative, qualitative, or property requirements",
            # },
            "Risk_mitigation_strategy": {
                "properties": ["method", "criteria", "purpose", "name", "description"],
                "description": "means to manage, mitigate or avoid a risk: reviews, safety nets, cross-validation, fallback architectures",
            },

            # "Metric_or_kpi": {
            #     "properties": ["name", "description"],
            #     "description": "relevant metrics or indicator such as: error, loss, generalization, among others",
            # },
            
            # # Governance & Metadata
            # "Organization": {
            #     "properties": ["complete name", "acronym", "purpose", "name", "description"],
            #     "description": "including but not limited to: begulatory bodies, OEM, suppliers, or certification authorities",
            # },
            # "Document": {
            #     "properties":["title", "publication_date", "version", "name", "description"],
            #     "description": "Standard recommendations, system design specs, verification report, and other relevant documents",
            # },
        }
        self.nodes = list(self.nodes_schema.keys())
        self.node_properties = True

        # Relations
        self.relations_schema = {
            # Document & Governance Relationships
            # ("Organization", "PUBLISHES", "Document"),
            # "PUBLISHES": {
            #     "heads": ["Organization"],
            #     "tails": ["Document"],
            #     "properties": ["date", "name", "description"],
            #     "description": "A Organization is responsible for issuing, preparing, publishing a Document"
            # },
            # # ("Document", "REFERENCES", "Document"),
            # "REFERENCES": {
            #     "heads": ["Document"],
            #     "tails": ["Document"],
            #     "properties": ["date", "name", "description"],
            #     "description": "A Document might reference or cite another Document"
            # },

            # Model Composition
            # ("Aiml_system", "INCLUDES", "Aiml_system"),
            # ("Aiml_element", "INCLUDES", "Aiml_element"),
            "INCLUDES": {
                "heads": ["Aiml_element"],
                "tails": ["Aiml_element"],
                "properties": ["name", "description"],
                "description": ""
            },
            # ("Aiml_element", "CONSUMES", "Aiml_resource"),
            "CONSUMES": {
                "heads": ["Aiml_element"],
                "tails": ["Aiml_resource"],
                "properties": ["name", "description"],
                "description": ""
            },
            # ("Aiml_lifecycle_aspect", "GOVERNS", "Aiml_element"),
            "GOVERNS": {
                "heads": ["Aiml_lifecycle_aspect"],
                "tails": ["Aiml_element"],
                "properties": ["name", "description"],
                "description": "An AI/ML element, system or product is developed, maintained, deployed, etc."
            },

            # Risk & Mitigation Lifecycle            
            # ("Aiml_element", "SUBJECT_TO", "Risk_factor"),
            # ("Aiml_resource", "SUBJECT_TO", "Risk_factor"),
            # ("Aiml_lifecycle_aspect", "SUBJECT_TO", "Risk_factor"),
            "SUBJECT_TO": {
                "heads": ["Aiml_element", "Aiml_resource", "Aiml_lifecycle_aspect"],
                "tails": ["Risk_factor"],
                "properties": ["name", "description"],
                "description": "All aspects in AI/ML are associated to risks"
            },
            # ("Risk_factor", "MITIGATED_BY", "Risk_mitigation_strategy")
            "MITIGATED_BY": {
                "heads": ["Risk_factor"],
                "tails": ["Risk_mitigation_strategy"],
                "properties": ["name", "description"],
                "description": "Mitigation strategies are recomended to mitigate risks associated to AI/ML"
            },
        }
        self.relations = []
        for name in self.relations_schema:
            for head in self.relations_schema[name]["heads"]:
                for tail in self.relations_schema[name]["tails"]:
                    self.relations.append((head, name, tail))
        self.relationship_properties = True

        # ontology description
        ontology_context = (
            "This ontology reflects relevant aspects in the development, maintenance or deployment of"
            "products and components based on or using Artificial Inteligence and Machine Learning,"
            "with focus on recomendation compliance and safety, in the context of civil aviation."
        )

        node_descriptions = "The node schemas (properties and descriptions): \n"
        for name in self.nodes_schema:
            node_descriptions += "\t" + name + ":\n"
            node_descriptions += "\t * description: " + self.nodes_schema[name]["description"] + "\n"
            node_descriptions += "\t * properties: " + ", ".join(self.nodes_schema[name]["properties"]) + "\n"
            
        relation_descriptions = "The relation schemas (properties and descriptions): \n"
        for name in self.relations_schema:
            relation_descriptions += "\t" + name + ":\n"
            relation_descriptions += "\t * description: " + self.relations_schema[name]["description"] + "\n"
            relation_descriptions += "\t * properties: " + ", ".join(self.relations_schema[name]["properties"]) + "\n"

        self.description = "\n\n".join([ontology_context, node_descriptions, relation_descriptions])

class SafefyOntology:
    pass

class DocumentationOntology:
    pass