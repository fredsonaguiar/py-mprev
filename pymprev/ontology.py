
# class Node:
#     def __init__(self, name:str, properties:list[str], description:str):
#         self.name = name
#         self.properties = properties
#         self.description = description


# class Relation:
#     def __init__(self, name:str, head:Node, tail:Node, properties:list[str], description:str):
#         self.name = name
#         self.properties = properties
#         self.description = description

#         self.head = head.name
#         self.tail = tail.name


class Ontology:
    def __init__(self):
        # Nodes
        self.nodes_schema = {
            # AI/ML Aspects
            # "AIML_System": {
            #     "properties": ["name", "description"],
            #     "description": "systems and subsystems, models, tools and products based on or using AI/ML",
            # },
            "AIML_Element": {
                "properties": ["definition", "purpose", "name", "description"],
                "description": "elements or components. systems and subsystems, models, tools and products based on or using AI/ML: Neural Nets, Models, etc",
            },
            "AIML_Resource": {
                "properties": ["definition", "name", "description"],
                "description": "training sets, generators, accelerators (GPU/FPGA), libraries, parameters, hyperparameters",
            },
            "AIML_Lifecycle_Aspect": {
                "properties": ["key_deliverable", "definition", "name", "description"],
                "description": "Aspects related to AI/ML development and maintenance: implementation, training, validation, deployment, etc",
            },

            # 
            "Risk_Factor": {
                "properties": ["name", "description"],
                "description": "Risks system or component may be subject to: failures, biases, dataset distribution shifts, operational hazards, among others",
            },
            # "Risk_Requirement": {
            #     "properties": ["name", "description"],
            #     "description": "Requirements a AI/ML system must comply with: functional, quantitative, qualitative, or property requirements",
            # },
            "Risk_Mitigation_Strategy": {
                "properties": ["method", "criteria", "purpose", "name", "description"],
                "description": "means to manage, mitigate or avoid a risk: reviews, safety nets, cross-validation, fallback architectures",
            },

            # "Metric_or_KPI": {
            #     "properties": ["name", "description"],
            #     "description": "relevant metrics or indicator such as: error, loss, generalization, among others",
            # },
            
            # Governance & Metadata
            "Organization": {
                "properties": ["complete name", "acronym", "purpose", "name", "description"],
                "description": "including but not limited to: begulatory bodies, OEM, suppliers, or certification authorities",
            },
            "Document": {
                "properties":["title", "publication_date", "version", "name", "description"],
                "description": "Standard recommendations, system design specs, verification report, and other relevant documents",
            },
        }
        self.nodes = list(self.nodes_schema.keys())
        self.node_properties = True

        # Relations
        self.relations_schema = {
            # Document & Governance Relationships
            # ("Organization", "PUBLISHES", "Document"),
            "PUBLISHES": {
                "heads": ["Organization"],
                "tails": ["Document"],
                "properties": ["date", "name", "description"],
                "description": "A Organization is responsible for issuing, preparing, publishing a Document"
            },
            # ("Document", "REFERENCES", "Document"),
            "REFERENCES": {
                "heads": ["Document"],
                "tails": ["Document"],
                "properties": ["date", "name", "description"],
                "description": "A Document might reference or cite another Document"
            },

            # Model Composition
            # ("AIML_System", "INCLUDES", "AIML_System"),
            # ("AIML_Element", "INCLUDES", "AIML_Element"),
            "INCLUDES": {
                "heads": ["AIML_Element"],
                "tails": ["AIML_Element"],
                "properties": ["name", "description"],
                "description": ""
            },
            # ("AIML_Element", "CONSUMES", "AIML_Resource"),
            "CONSUMES": {
                "heads": ["AIML_Element"],
                "tails": ["AIML_Resource"],
                "properties": ["name", "description"],
                "description": ""
            },
            # ("AIML_Lifecycle_Aspect", "GOVERNS", "AIML_Element"),
            "GOVERNS": {
                "heads": ["AIML_Lifecycle_Aspect"],
                "tails": ["AIML_Element"],
                "properties": ["name", "description"],
                "description": "An AI/ML element, system or product is developed, maintained, deployed, etc."
            },

            # Risk & Mitigation Lifecycle            
            # ("AIML_Element", "SUBJECT_TO", "Risk_Factor"),
            # ("AIML_Resource", "SUBJECT_TO", "Risk_Factor"),
            # ("AIML_Lifecycle_Aspect", "SUBJECT_TO", "Risk_Factor"),
            "SUBJECT_TO": {
                "heads": ["AIML_Element", "AIML_Resource", "AIML_Lifecycle_Aspect"],
                "tails": ["Risk_Factor"],
                "properties": ["name", "description"],
                "description": "All aspects in AI/ML are associated to risks"
            },
            # ("Risk_Factor", "MITIGATED_BY", "Risk_Mitigation_Strategy")
            "MITIGATED_BY": {
                "heads": ["Risk_Factor"],
                "tails": ["Risk_Mitigation_Strategy"],
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

