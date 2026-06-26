# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle


class CrystalBatch:
    def __init__(self):
        self.cart_coords = None
        self.frac_coords = None
        self.lattices = None
        self.num_atoms = None
        self.lengths = None
        self.lengths_scaled = None
        self.angles = None
        self.angles_radians = None
        self.atom_types = None
        self.pos = None
        self.token_idx = None
        self.batch = None
        self.y = None
        self.num_nodes = None
        self.mace_features = None
        self.mask = None
        self.zs = None
        self.means = None
        self.stds = None
        self.num_graphs = 0

    def add(self, **kwargs):
        for key, tensor in kwargs.items():
            if not isinstance(key, str):
                raise TypeError(f"Key must be a string, got {type(key).__name__}.")
            if not isinstance(tensor, paddle.Tensor):
                raise TypeError(
                    f"Value must be a paddle.Tensor, got {type(tensor).__name__}."
                )
            if hasattr(self, key):
                raise KeyError(f"Attribute '{key}' already exists in the batch.")
            setattr(self, key, tensor)

    def update(self, allow_reshape=False, **kwargs):
        for key, tensor in kwargs.items():
            if not isinstance(key, str):
                raise TypeError(f"Key must be a string, got {type(key).__name__}.")
            if not hasattr(self, key):
                raise KeyError(f"Attribute '{key}' not found in the batch.")
            if not isinstance(tensor, paddle.Tensor):
                raise TypeError(
                    f"Value must be a paddle.Tensor, got {type(tensor).__name__}."
                )

            existing = getattr(self, key)
            if tensor.shape != existing.shape and not allow_reshape:
                raise ValueError(
                    f"Shape mismatch for '{key}': existing {tuple(existing.shape)}, new {tuple(tensor.shape)}."
                )
            setattr(self, key, tensor)

    def remove(self, *keys):
        if len(keys) == 0:
            raise ValueError("At least one key must be provided to remove().")
        for key in keys:
            if not isinstance(key, str):
                raise TypeError(f"Key must be a string, got {type(key).__name__}.")
            if not hasattr(self, key):
                raise KeyError(f"Attribute '{key}' not found in the batch.")
            delattr(self, key)

    def to(self, device):
        for attr_name in dir(self):
            if attr_name.startswith('_') or callable(getattr(self, attr_name)):
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, paddle.Tensor):
                setattr(self, attr_name, attr.cuda() if device == 'gpu' else attr.cpu())
            elif isinstance(attr, dict):
                for k, v in attr.items():
                    if isinstance(v, paddle.Tensor):
                        attr[k] = v.cuda() if device == 'gpu' else v.cpu()
        return self

    def clone(self):
        cloned = CrystalBatch()
        for attr_name in dir(self):
            if attr_name.startswith('_') or callable(getattr(self, attr_name)):
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, paddle.Tensor):
                setattr(cloned, attr_name, attr.clone())
            elif isinstance(attr, dict):
                setattr(cloned, attr_name, {k: v.clone() if isinstance(v, paddle.Tensor) else v for k, v in attr.items()})
            else:
                setattr(cloned, attr_name, attr)
        return cloned

    def _split_by_batch_index(self):
        if self.batch is None or self.num_atoms is None:
            raise ValueError("batch and num_atoms must be set to split the batch")
        
        structures = []
        batch_indices = self.batch.cpu().numpy()
        num_graphs = int(batch_indices.max()) + 1 if len(batch_indices) > 0 else 0
        
        for i in range(num_graphs):
            mask = batch_indices == i
            structure_data = {}
            
            if self.atom_types is not None:
                structure_data['atom_types'] = self.atom_types[mask]
            if self.frac_coords is not None:
                structure_data['frac_coords'] = self.frac_coords[mask]
            if self.cart_coords is not None:
                structure_data['cart_coords'] = self.cart_coords[mask]
            if self.lattices is not None:
                if self.lattices.ndim == 3:
                    structure_data['lattices'] = self.lattices[i]
                else:
                    structure_data['lattices'] = self.lattices
            
            structures.append(structure_data)
        
        return structures

    def to_atoms(self, frac_coords=True):
        try:
            from ase import Atoms
        except ImportError:
            raise ImportError("ASE is required for to_atoms(). Install it with: pip install ase")
        
        if self.atom_types is None or self.lattices is None:
            raise ValueError("atom_types and lattices must be set")
        
        structures = self._split_by_batch_index()
        atoms_list = []
        
        for struct_data in structures:
            atom_types_np = struct_data['atom_types'].cpu().numpy()
            lattice_np = struct_data['lattices'].cpu().numpy()
            
            if lattice_np.ndim == 3:
                lattice_np = lattice_np.squeeze(0)
            
            atoms = Atoms(
                numbers=atom_types_np,
                cell=lattice_np,
                pbc=True,
            )
            
            if frac_coords and 'frac_coords' in struct_data:
                positions = struct_data['frac_coords'].cpu().numpy()
                atoms.set_scaled_positions(positions)
            elif 'cart_coords' in struct_data:
                positions = struct_data['cart_coords'].cpu().numpy()
                atoms.set_positions(positions)
            
            atoms_list.append(atoms)
        
        return atoms_list

    def to_structures(self, frac_coords=True):
        try:
            from pymatgen.core import Lattice, Structure, Element
        except ImportError:
            raise ImportError("Pymatgen is required for to_structures(). Install it with: pip install pymatgen")
        
        if self.atom_types is None or self.lattices is None:
            raise ValueError("atom_types and lattices must be set")
        
        structures = self._split_by_batch_index()
        structure_list = []
        
        for struct_data in structures:
            atom_types_int = struct_data['atom_types'].cpu().numpy().tolist()
            if isinstance(atom_types_int[0], list):
                atom_types_int = [item for sublist in atom_types_int for item in sublist]
            atom_types_symbols = [Element.from_Z(int(z)).symbol for z in atom_types_int]
            
            lattice_np = struct_data['lattices'].cpu().numpy()
            
            if lattice_np.ndim == 3:
                lattice_np = lattice_np.squeeze(0)

            lattice = Lattice(lattice_np)
            
            if frac_coords and 'frac_coords' in struct_data:
                coords = struct_data['frac_coords'].cpu().numpy()
                structure = Structure(
                    lattice=Lattice.from_parameters(*lattice.parameters),
                    species=atom_types_symbols,
                    coords=coords,
                    coords_are_cartesian=False,
                )
            elif 'cart_coords' in struct_data:
                coords = struct_data['cart_coords'].cpu().numpy()
                structure = Structure(
                    lattice=Lattice.from_parameters(*lattice.parameters),
                    species=atom_types_symbols,
                    coords=coords,
                    coords_are_cartesian=True,
                )
            else:
                raise ValueError("Either frac_coords or cart_coords must be available")
            
            structure_list.append(structure)
        
        return structure_list

    @classmethod
    def from_data_list(cls, data_list):
        if not data_list:
            return cls()
        
        batch = cls()
        batch.num_graphs = len(data_list)
        
        batch_indices = []
        
        for graph_idx, data in enumerate(data_list):
            num_nodes = data.get('num_atoms', 0)
            if isinstance(num_nodes, paddle.Tensor):
                num_nodes = int(num_nodes.item())
            batch_indices.extend([graph_idx] * num_nodes)
        
        batch.batch = paddle.to_tensor(batch_indices, dtype='int64')
        
        for key in ['atom_types', 'frac_coords', 'cart_coords', 'pos', 'token_idx']:
            tensors = [data.get(key) for data in data_list if key in data]
            if tensors:
                batch.__dict__[key] = paddle.concat(tensors, axis=0)
        
        for key in ['lattices', 'num_atoms', 'lengths', 'lengths_scaled', 'angles', 'angles_radians']:
            tensors = [data.get(key) for data in data_list if key in data]
            if tensors:
                if key == 'lattices':
                    stacked = []
                    for t in tensors:
                        if t.ndim == 2:
                            t = t.unsqueeze(0)
                        stacked.append(t)
                    batch.__dict__[key] = paddle.concat(stacked, axis=0)
                elif key == 'num_atoms':
                    stacked = []
                    for t in tensors:
                        if t.ndim == 0:
                            t = t.unsqueeze(0)
                        stacked.append(t)
                    batch.__dict__[key] = paddle.concat(stacked, axis=0)
                else:
                    stacked = []
                    for t in tensors:
                        if t.ndim == 1:
                            t = t.unsqueeze(0)
                        stacked.append(t)
                    batch.__dict__[key] = paddle.concat(stacked, axis=0)
        
        if 'y' in data_list[0]:
            y_dict = {}
            for key in data_list[0]['y'].keys():
                y_values = [data['y'][key] for data in data_list]
                if isinstance(y_values[0], paddle.Tensor):
                    y_dict[key] = paddle.concat(y_values, axis=0)
                else:
                    y_dict[key] = y_values
            batch.y = y_dict
        
        num_nodes_list = []
        for data in data_list:
            num = data.get('num_atoms', 0)
            if isinstance(num, paddle.Tensor):
                num = int(num.item())
            num_nodes_list.append(num)
        batch.num_nodes = paddle.to_tensor(num_nodes_list, dtype='int64')
        
        batch.mask = paddle.ones([len(data_list), max(num_nodes_list)], dtype='bool')
        for i, num in enumerate(num_nodes_list):
            if num < max(num_nodes_list):
                batch.mask[i, num:] = False
        
        return batch

    @classmethod
    def collate(cls, data_list):
        return cls.from_data_list(data_list)
    
    def repeat(self, num_repeats):
        if num_repeats <= 1:
            return self
        
        repeated_batch = CrystalBatch()
        repeated_batch.num_graphs = self.num_graphs * num_repeats
        
        if self.batch is not None:
            batch_indices = []
            for i in range(num_repeats):
                offset = i * self.num_graphs
                batch_indices.append(self.batch + offset)
            repeated_batch.batch = paddle.concat(batch_indices, axis=0)
        
        for key in ['atom_types', 'frac_coords', 'cart_coords', 'pos', 'token_idx']:
            attr = self.__dict__.get(key)
            if attr is not None:
                repeated_batch.__dict__[key] = paddle.tile(attr, [num_repeats] + [1] * (attr.ndim - 1))
        
        for key in ['lattices', 'num_atoms', 'lengths', 'lengths_scaled', 'angles', 'angles_radians']:
            attr = self.__dict__.get(key)
            if attr is not None:
                repeated_batch.__dict__[key] = paddle.tile(attr, [num_repeats] + [1] * (attr.ndim - 1))
        
        if self.y is not None:
            repeated_y = {}
            for key, value in self.y.items():
                if isinstance(value, paddle.Tensor):
                    repeated_y[key] = paddle.tile(value, [num_repeats] + [1] * (value.ndim - 1))
                else:
                    repeated_y[key] = value * num_repeats
            repeated_batch.y = repeated_y
        
        if self.num_nodes is not None:
            repeated_batch.num_nodes = self.num_nodes * num_repeats
        
        if self.mask is not None:
            repeated_batch.mask = paddle.tile(self.mask, [num_repeats, 1])
        
        return repeated_batch


def create_empty_batch(num_atoms, device='cpu', atom_types=None):
    data_list = []
    for i, n in enumerate(num_atoms):
        data = {
            'pos': paddle.empty([n, 3]),
            'atom_types': (
                paddle.empty([n], dtype='int64')
                if atom_types is None
                else paddle.to_tensor(atom_types[i], dtype='int64')
            ),
            'frac_coords': paddle.empty([n, 3]),
            'cart_coords': paddle.empty([n, 3]),
            'lattices': paddle.empty([1, 3, 3]),
            'num_atoms': paddle.to_tensor(n, dtype='int64'),
            'lengths': paddle.empty([1, 3]),
            'lengths_scaled': paddle.empty([1, 3]),
            'angles': paddle.empty([1, 3]),
            'angles_radians': paddle.empty([1, 3]),
            'token_idx': paddle.arange(n, dtype='int64'),
        }
        data_list.append(data)
    
    batch = CrystalBatch.from_data_list(data_list)
    if device == 'gpu':
        batch = batch.to('gpu')
    return batch
