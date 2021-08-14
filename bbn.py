# by Atharva Manjrekar

# Import relevant libraries
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

# Outline the Bayesian Network nodes 
# I created my own random probabilities for each variable since none were provided 
exercise = BbnNode(Variable(0, 'exercise', ['yes', 'no']), [0.5, 0.5])
diet = BbnNode(Variable(1, 'diet', ['healthy', 'unhealthy']), [0.5, 0.5])
heart_disease = BbnNode(Variable(2, 'heart_disease', ['yes', 'no']), [0.7, 0.3, 0.9, 0.1, 0.1, 0.9, 0.3, 0.7])
chest_pain = BbnNode(Variable(3, 'chest_pain', ['yes', 'no']), [0.9, 0.1, 0.2, 0.8])
blood_pressue = BbnNode(Variable(4, 'blood_pressure', ['high', 'low']), [0.8, 0.2, 0.7, 0.3])

# This will outline the nodes as directed by the diagram to create a tree
bbn = Bbn().add_node(exercise).add_node(diet).add_node(heart_disease).add_node(chest_pain)\
    .add_node(blood_pressue).add_edge(Edge(exercise, heart_disease, EdgeType.DIRECTED))\
        .add_edge(Edge(diet, heart_disease, EdgeType.DIRECTED)).add_edge(Edge(heart_disease, chest_pain, EdgeType.DIRECTED))\
            .add_edge(Edge(heart_disease, blood_pressue, EdgeType.DIRECTED))

join_tree = InferenceController.apply(bbn)

# Here we can use the EvidenceBuilder to insert new evidence
# This will allow us to answer relevant questions pertaining to the probability of one variable given evidence of other variables
ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name('exercise')) \
    .with_evidence('yes', 1.0) \
    .build()

ev2 = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name('diet')) \
    .with_evidence('unhealthy', 1.0) \
    .build()

ev3 = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name('chest_pain')) \
    .with_evidence('yes', 1.0) \
    .build()

join_tree.set_observation(ev)
join_tree.set_observation(ev2)
join_tree.set_observation(ev3)

# Here we will print the marginal probabilities
for node in join_tree.get_bbn_nodes():
    potential = join_tree.get_bbn_potential(node)
    print(node)
    print(potential)
    print('---------------------')