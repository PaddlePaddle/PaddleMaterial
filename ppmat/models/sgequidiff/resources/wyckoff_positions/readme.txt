The file, 'clean_wyckoffs_in_asu_v6.json', has the following structure:

{ space_group_number:
	{'ordered_wyckoff_letters': ['a', 'b', ...],
	 'hall_number': integer,
	 'a': {  # 0d shape (1, 3)
	    'vertices': [x, y, z],
	    'dim': '0',
	    'volumes': [0.0],
	    }
	 'b': {  # 1d shape (n_segments, 2, 3)
	    'vertices': [segment1, …],
	    'dim': '1',
	    'volumes': [segment1 volume, ...],
	 }
	 'c': {  # 2d shape (n_faces, n_face_vertices, 3) where 'n_face_vertices' varies across faces
	    'vertices': [face1, …],
	    'dim': '2',
	    'plane_coefficients': [face1_plane, …],
	    'volumes': [face1 volume, ...],
	 }
	 'd': {  # 3d shape (n_vertices, 3)
	    'vertices': [vertex1, …],
	    'dim': '3',
        'volumes': [ASU volume],
	 }
	}
}

segment := [vertex1, vertex2]
face := [vertex1, …]
face_plane := [a, b, c, d]  # ax+by+cz=d
vertex := [x, y, z]