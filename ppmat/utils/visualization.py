import os

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rdkit
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import RDKFingerprint
from rdkit.Geometry import Point3D

from ppmat.utils import logger


class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos, output_dir):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos
        self.result_path = os.path.join(output_dir, f"graph/")

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        atom_decoder = self.dataset_infos.atom_decoder
        mol = Chem.RWMol()
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx
        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            logger.info("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(
        self, path: str, molecules: list, num_molecules_to_visualize: int
    ):
        if not os.path.exists(path):
            os.makedirs(path)
        logger.message(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            logger.message(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, "molecule_{}.png".format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            try:
                Draw.MolToFile(mol, file_path)
            except rdkit.Chem.KekulizeException:
                logger.info("Can't kekulize molecule")

    def visualizeNmr(
        self,
        batch_id,
        molecules: list,
        molecules_true,
        num_molecules_to_visualize: int,
        log="graph",
    ):
        path = os.path.join(self.result_path, f"batch_{batch_id}_predicted")
        path_true = os.path.join(self.result_path, f"batch_{batch_id}_true")
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path_true):
            os.makedirs(path_true)
        logger.message(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            logger.info(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, "molecule_{}.png".format(i))
            file_path_true = os.path.join(path_true, "molecule_{}.png".format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            mol_true = self.mol_from_graphs(
                molecules_true[i][0].numpy(), molecules_true[i][1].numpy()
            )
            try:
                Draw.MolToFile(mol, file_path)
                Draw.MolToFile(mol_true, file_path_true)
                fp1 = RDKFingerprint(mol)
                fp2 = RDKFingerprint(mol_true)
                similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                logger.info(f"Tanimoto相似度: {similarity}")
            except rdkit.Chem.KekulizeException:
                logger.info("Can't kekulize molecule")

    def visualize_chain(self, batch_id, i, nodes_list, adjacency_matrix):
        path = os.path.join(self.result_path, f"chain/molecule_{batch_id}_{i}")
        os.makedirs(path, exist_ok=True)
        RDLogger.DisableLog("rdApp.*")
        mols = [
            self.mol_from_graphs(nodes_list[i], adjacency_matrix[i])
            for i in range(nodes_list.shape[0])
        ]
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)
        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))
        for i, mol in enumerate(mols):
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))
        save_paths = []
        num_frams = nodes_list.shape[0]
        for frame in range(num_frams):
            file_name = os.path.join(path, "fram_{}.png".format(frame))
            Draw.MolToFile(
                mols[frame], file_name, size=(300, 300), legend=f"Frame {frame}"
            )
            save_paths.append(file_name)
        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(
            os.path.dirname(path), "{}.gif".format(path.split("/")[-1])
        )
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
            img.save(
                os.path.join(path, "{}_grid_image.png".format(path.split("/")[-1]))
            )
        except Chem.rdchem.KekulizeException:
            logger.info("Can't kekulize molecule")
        return mols


class NonMolecularVisualization:
    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])
        rows, cols = np.where(adjacency_matrix >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(
                edge[0], edge[1], color=float(edge_type), weight=3 * edge_type
            )
        return graph

    def visualize_non_molecule(
        self, graph, pos, path, iterations=100, node_size=100, largest_component=False
    ):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m
        plt.figure()
        nx.draw(
            graph,
            pos,
            font_size=5,
            node_size=node_size,
            with_labels=False,
            node_color=U[:, 1],
            cmap=plt.cm.coolwarm,
            vmin=vmin,
            vmax=vmax,
            edge_color="grey",
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(
        self, path: str, graphs: list, num_graphs_to_visualize: int
    ):
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, "graph_{}.png".format(i))
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
            im = plt.imread(file_path)  # noqa

    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        graphs = [
            self.to_networkx(nodes_list[i], adjacency_matrix[i])
            for i in range(nodes_list.shape[0])
        ]
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)
        save_paths = []
        num_frams = nodes_list.shape[0]
        for frame in range(num_frams):
            file_name = os.path.join(path, "fram_{}.png".format(frame))
            self.visualize_non_molecule(
                graph=graphs[frame], pos=final_pos, path=file_name
            )
            save_paths.append(file_name)
        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(
            os.path.dirname(path), "{}.gif".format(path.split("/")[-1])
        )
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)
