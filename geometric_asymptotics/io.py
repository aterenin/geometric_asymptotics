import numpy as np
import scipy
import gmshparser
from pathlib import Path
from geometric_kernels.spaces.mesh import Mesh
from geometric_kernels.spaces.graph import Graph

def assemble_2d_laplacian(path: Path):
    gmsh_mesh = gmshparser.parse(path)
    nodes = [node for entity in gmsh_mesh.get_node_entities() for node in entity.get_nodes()]
    idxs = {node.get_tag(): idx for idx,node in enumerate(nodes)}
    elements = [element.get_connectivity() for entity in gmsh_mesh.get_element_entities() if entity.get_element_type()==1 for element in entity.get_elements()]

    coordinates = np.stack([np.array(node.get_coordinates()[:2]) for node in nodes])

    def weight(x1,x2):
        return np.exp(-np.linalg.norm(np.array(x1) - np.array(x2))/2)

    N = len(elements)
    i = [idxs[element[0]] for element in elements]
    j = [idxs[element[1]] for element in elements]
    v = [weight(nodes[idxs[element[0]]].get_coordinates(), nodes[idxs[element[1]]].get_coordinates()) for element in elements]

    off_diag = scipy.sparse.coo_array((v,(i,j)), shape=(N,N))

    return (coordinates, off_diag + off_diag.T)

def load_mesh(path: Path):
    mesh = Mesh.load_mesh(str(path))
    coordinates = np.array(mesh.vertices)
    return (mesh, coordinates)

def load_graph(path: Path):
    coordinates, adjacency_matrix = assemble_2d_laplacian(path)
    # temporarily switch to dense computations due to upstream sparse matrix shape check bug
    adjacency_matrix = adjacency_matrix.todense() # TODO: remove once fixed
    graph = Graph(adjacency_matrix)
    return (graph, coordinates)

def load_space(source: str):
    path = Path(source)
    if path.suffix == ".msh":
        return load_graph(path)
    elif path.suffix == ".obj":
        return load_mesh(path)
    