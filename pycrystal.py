# -- coding:utf-8 --

#########################################################################
#   Copyright (C) 2021, Tang Kewei <tangkeweisirius.hotmail.com>
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License v3.0 as 
#   published by the Free Software Foundation.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License v3.0
#   along with this program; if not, see <http://www.gnu.org/licenses/>
#########################################################################


"""a module for crystal operation (especially 2D crystals)"""

___author___ = 'Tang Kewei'


import numpy as np
import os.path
import copy
import math
from typing import Dict, List, Tuple



############################################################################################
# exceptions & checkers:
############################################################################################
class ArgError(Exception):
    def __init__(self, argName: str, des=""):
        super().__init__(self)
        self._arg_name = argName
        self._des = des

    def __str__(self):
        if self._des:
            msg = 'argument: ' + self._arg_name + ' is invalid! ' + '(' + self._des +')'
        else:
            msg = 'argument: ' + self._arg_name + ' is invalid!'
        return msg


class UnSupportedError(Exception):
    def __init__(self, arg: str):
        super().__init__(self)
        self._arg = arg

    def __str__(self):
        return self._arg + ' is not supported!'


class NumpyShapeError(Exception):
    def __init__(self, expected_shape :Tuple[int], shape: Tuple[int]):
        super().__init__(self)
        self._expected_shape = expected_shape # type: tuple(int)
        self._shape = shape # type: tuple(int)

    def __str__(self):
        return "Wrong array shape: expected %s, got %s" % (str(self._expected_shape), str(self._shape))


def check_2darray(a: np.ndarray, i: int, j: int):
    """
    assert that an object is an np.ndarray with shape (i, j)

    :param a: the object to be asserted
    :param i: row count
    :param j: column count
    """
    assert(isinstance(a, np.ndarray))
    if a.shape != (i, j):
        raise NumpyShapeError((i, j), a.shape)


def check_1darray(a: np.ndarray, i: int):
    """
    assert that an object is an np.ndarray with shape (i,)

    :param a: the object to be asserted
    :param i: elements count
    """
    assert(isinstance(a, np.ndarray))
    if a.shape != (i,):
        raise NumpyShapeError((i,), a.shape)



###################################################################
# helper functions:
###################################################################
def _calc_transform(a: float, b: float, c: float) -> np.ndarray:
    """
    calc transform with Eular angle

    :param a: angle on axis 0
    :param b: angle on axis 1
    :param c: angle on axis 2

    :returns: transform
    """
    a = a * math.pi/180
    b = b * math.pi/180
    c = c * math.pi/180

    sin_a = math.sin(a)
    cos_a = math.cos(a)
    sin_b = math.sin(b)
    cos_b = math.cos(b)
    sin_c = math.sin(c)
    cos_c = math.cos(c)

    t0 = np.array([\
            [1,      0,     0],\
            [0,  cos_a, sin_a],\
            [0, -sin_a, cos_a]])
    t1 = np.array([\
            [ cos_b, 0, sin_b],\
            [     0, 1,     0],\
            [-sin_b, 0, cos_b]])
    t2 = np.array([\
            [ cos_c, sin_c, 0],\
            [-sin_c, cos_c, 0],\
            [     0,     0, 1]])

    return np.array(np.matmul(np.matmul(t0, t1), t2))



###################################################################################################
# Definition of Lattice class:
###################################################################################################

class Lattice:
    """
    Lattice with various operations (vectors are all row vectors)

    NOTE: since both the Cartesian and fractional atom positions can
    be accessed frequently, both of them are stored as member data. Make
    sure to call cart2frac or frac2cart when changes are made directly to one of them.
    """
    def __init__(self, sysname: str, bvs: np.ndarray,
                 apd: Dict[str, np.ndarray]=None, apd_f: Dict[str, np.ndarray]=None):
        self._sysname = sysname # type: str

        check_2darray(bvs, 3, 3)
        self._bvs = copy.deepcopy(bvs) # type: np.ndarray

        self._apd = {} # type: dict[str, np.ndarray]
        self._apd_f = {} # type: dict[str, np.ndarray]

        # raise exception if there are whitespace in keys (elements)
        # instead of implicitly correct it !!!
        if apd is not None:
            for e, atoms in apd.items():
                if e.strip() != e:
                    raise ArgError("apd", "chemical elements should contain NO whitespace!")
                if atoms.dtype != float:
                    raise ArgError("apd", "atom coordinates should be float numbers")
                self._apd[e] = copy.deepcopy(atoms)
            self._cart2frac()
        elif apd_f is not None:
            for e, atoms in apd_f.items():
                if e.strip() != e:
                    raise ArgError("apd_f", "chemical elements should contain NO whitespace!")
                if atoms.dtype != float:
                    raise ArgError("apd_f", "atom coordinates should be float numbers")
                self._apd_f[e] = copy.deepcopy(atoms)
            self._frac2cart()
        else:
            raise ArgError("apd, apd_f", des="can't BOTH be None!")


    def __str__(self):
        apd_str = ""
        for e, atoms in self._apd.items():
            apd_str += e + ":\n"
            apd_str += str(atoms) + "\n"

        apd_f_str = ""
        for e, atoms in self._apd_f.items():
            apd_f_str += e + ":\n"
            apd_f_str += str(atoms) + "\n"

        return str(self._sysname) + "\n" +\
               str(self._bvs) + "\n" +\
               "APD:\n" +\
               apd_str +\
               "fractional APD:\n" +\
               apd_f_str


    def confine_atoms_to_lattice(self):
        """
        put outside atoms back to the lattice by periodic repeating (-0.2 -> 0.8, e. g.)

        NOTE: DON'T use this if there are vaccum in the lattice (2D lattice) and
        the lattice atoms are actually shifted away from the lattice
        """
        for atoms in self._apd_f.values():
            for a in atoms: # implicit shallow copy
                for i in range(3):
                    if a[i] < 0.0:
                        a[i] += 1.0
                    elif a[i] >= 1.0:
                        a[i] -= 1.0
        
        self._frac2cart()


    def set_sysname(self, s: str):
        self._sysname = str(s)


    def get_sysname(self) -> str:
        return self._sysname


    def replace_bvs(self, bvs: np.ndarray):
        """
        replace basis vectors

        :param bvs: the new basis vectors

        NOTE: This will retain the APD_f! (and update APD)
        (see set_bvs)
        """
        check_2darray(bvs, 3, 3)

        self._bvs = copy.deepcopy(bvs)
        self._frac2cart()


    def set_bvs(self, bvs: np.ndarray):
        """
        re-set the basis vectors with 'bv'.

        NOTE: This will retain the APD! (and update APD_f)
        """
        check_2darray(bvs, 3, 3)

        self._bvs = copy.deepcopy(bvs)
        self._cart2frac()


    def get_bvs(self) -> np.ndarray:
        return copy.deepcopy(self._bvs)


    def lattice_lengths(self) -> np.ndarray:
        return np.array([ np.linalg.norm(self._bvs[i]) for i in range(3) ])


    def lattice_angles(self) -> np.ndarray:
        """
        get lattice angles (in degree)
        (alpha: <bvs[1], bvs[2]>, beta: <bvs[0], bvs[2]>, gamma: <bvs[0], bvs[1]>)

        :returns: alpha, beta, gamma
        """
        r = self.lattice_lengths()
        return np.array([
            ( math.acos(sum(self._bvs[1]*self._bvs[2]) / (r[1]*r[2])) )*180/math.pi,
            ( math.acos(sum(self._bvs[0]*self._bvs[2]) / (r[0]*r[2])) )*180/math.pi,
            ( math.acos(sum(self._bvs[0]*self._bvs[1]) / (r[0]*r[1])) )*180/math.pi ])


    def get_apd_f(self) -> Dict[str, np.ndarray]:
        """get fractional atom positions dict"""
        return copy.deepcopy(self._apd_f)


    def get_apd(self) -> Dict[str, np.ndarray]:
        """get Cartesian atom positions dict"""
        return copy.deepcopy(self._apd)


    def set_atom_position(self, e: str, i: int, ap: np.ndarray):
        """
        set i-th atom's Cartesian position

        :param e: atom species
        :param i: atom index
        :param ap: new atom position
        """
        check_1darray(ap, 3)
        self._apd[e][i] = copy.deepcopy(ap)
        self._cart2frac()


    def get_atom_position(self, e: str, i: int) -> np.ndarray:
        """
        get i-th atom's Cartesian position

        :param e: atom species
        :param i: atom index
        """
        return copy.deepcopy(self._apd[e][i])


    def set_atom_position_f(self, e: str, i: int, ap: np.ndarray):
        """
        set i-th atom's fractional position

        :param e: atom species
        :param i: atom index
        :param ap: new atom position
        """
        check_1darray(ap, 3)
        self._apd_f[e][i] = copy.deepcopy(ap)
        self._frac2cart()


    def get_atom_position_f(self, e: str, i: int) -> np.ndarray:
        """
        get i-th atom's fractional position

        :param e: atom species
        :param i: atom index
        """
        return copy.deepcopy(self._apd_f[e][i])


    def add_atom(self, e: str, ap: np.ndarray):
        """
        add an atom with cartesian position

        :param e: element
        :param ap: cartesian atom position

        :returns: the index of newly added atom
        """
        check_1darray(ap, 3)
        if e in self._apd.keys():
            self._apd[e] = np.row_stack((self._apd[e], ap))
            self._apd_f[e] = np.row_stack((self._apd_f[e],
                np.array(np.matmul(ap, np.matrix(self._bvs).I)) ))
            n = self._apd[e].shape[0]
        else:
            self._apd[e] = np.array([ap,]);
            self._apd_f[e] = np.array(np.matmul(ap, np.matrix(self._bvs).I))
            n = 0

        return n


    def add_atom_f(self, e: str, ap: np.ndarray) -> int:
        """
        add an atom with fractional position

        :param e: element
        :param ap: fractional atom position

        :returns: the index of newly added atom
        """
        check_1darray(ap, 3)
        if e in self._apd_f.keys():
            self._apd_f[e] = np.row_stack((self._apd_f[e], ap))
            self._apd[e] = np.row_stack((self._apd[e],
                np.array(np.matmul(ap, self._bvs )) ))
            n = self._apd_f[e].shape[0]
        else:
            self._apd_f[e] = np.array([ap,]);
            self._apd[e] = np.array(np.matmul(ap, self._bvs ))
            n = 0

        return n


    def delete_atom(self, e: str, i: int):
        """
        delete an atom

        :param e: atom species
        :param i: index
        """
        if e not in self._apd.keys() or not (0 <= i < self._apd[e].shape[0]):
            raise Exception("specified atom (%s, %d) does not exist!" % (e, i))

        self._apd[e] = np.delete(self._apd[e], i, axis=0)
        self._apd_f[e] = np.delete(self._apd_f[e], i, axis=0)

        # delete the element if there's no atoms of it
        if len(self._apd[e]) == 0:
            self._apd.pop(e)
            self._apd_f.pop(e)


    def _frac2cart(self):
        """update the fractional coordinates of atoms accroding to the Cartesian coordinates"""
        self._apd.clear()
        for e, atoms in self._apd_f.items():
            self._apd[e] = np.array(np.matmul(atoms, self._bvs))


    def _cart2frac(self):
        """update the Cartesian coordinates of atoms accroding to the fractional coordinates"""
        self._apd_f.clear()
        for e, atomPos in self._apd.items():
            self._apd_f[e] = np.array(np.matmul(atomPos, np.matrix(self._bvs).I))


    def atom_species(self) -> List[str]:
        return list(self._apd.keys())


    def atom_num_dict(self) -> Dict[str, int]:
        """get atom number dict (indexed by elements)"""
        andict = {}
        for e, atoms in self._apd.items():
            andict[e] = len(atoms)
        return andict

    
    def atom_num(self, e: str) -> int:
        """
        get the count of atoms of an atom species

        :param e: atom species
        :returns: the count the `e` atoms
        """
        return self._apd[e].shape[0]


    def total_atom_num(self) -> int:
        t = 0
        for atoms in self._apd.values():
            t += atoms.shape[0]
        return t

    
    def atom_distance(self, e0: str, i0: int, e1: str, i1: int) -> float:
        """
        get the 'closest' cartesian distance between two atoms

        :param e0: atom species of atom 0
        :param e1: atom species of atom 1
        :param i0: index of atom 0
        :param i1: index of atom 1

        :returns: the cartesian distance between atom 0 and atom 1
        """
        if e0 == e1 and i0 == i1:
            raise ArgError("e0, i0, e1, e1", "the two atoms must not be the same one!")

        p0 = self.get_atom_position(e0, i0) 
        p1 = self.get_atom_position(e1, i1)
        return min([ np.linalg.norm(p1 + np.array(np.matmul(np.array([i, j, k]), self._bvs)) - p0) 
                     for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])


    def truncate(self):
        """
        delete outside atoms
        """
        for e, n in self.atom_num_dict().items():
            for i in range(n-1, -1, -1):
                outside=False
                for x in self.get_atom_position_f(e, i):
                    if not (0 <= x < 1):
                        outside=True
                        break
                if outside:
                    self.delete_atom(e, i)


    def eliminate_overlap(self, crit: float):
        """
        delete overlapping atoms (this can be CPU-consuming)

        :param crit: maximum inter-atom distance (cartesian) to be considered as overlap
        """
        e_list = self.atom_species()
        en = len(e_list)
        for q in range(en-1, -1, -1):
            e_q = e_list[q] # q-th element
            n = self.atom_num_dict()[e_q] # number of atoms of q-th element
            for i in range(n-1, -1, -1):
                flag = False # deleted or not
                # same element atoms check:
                for j in range(i):
                    if self.atom_distance(e_q, i, e_q, j) <= crit:
                        # print("same element atoms removal")
                        # print(e_q, i)
                        # print(self.get_atom_position(e_q, i))
                        # print(e_q, j)
                        # print(self.get_atom_position(e_q, j))
                        # print("dist: " + str(self.atom_distance(e_q, i, e_q, j)))
                        self.delete_atom(e_q, i)
                        flag = True
                        break
                if flag:
                    continue

                # inter-elements atoms check
                for u in range(q):
                    e_u = e_list[u] # m-th element
                    m = self.atom_num_dict()[e_u] # numbre of atoms of m-th element
                    for k in range(m):
                        if self.atom_distance(e_q, i, e_u, k) <= crit:
                            self.delete_atom(e_q, i)
                            flag = True
                            break
                    if flag:
                        break

                if flag:
                    continue


    def reshape(self, new_shape: np.ndarray, crit: float=None):
        """
        reshape the lattice, fill atoms into the new shape
        NOTE: this method directly changes the shape of the cell, which can
              easily damage the periodicity. Use it use caution (see transform)

        :param new_shape: 3x3 array to specify a new lattice shape (BVs)
        :param crit: maximum inter-atom distance (cartesian) to be considered as overlap
        """
        # translate the box to the local BVs
        # so that we can calc the lattice range easily
        box_t = np.array(np.matmul(new_shape, np.matrix(self._bvs).I))

        ns0 = ns1 = ns2 = 0
        a = min(box_t[:, 0])
        b = min(box_t[:, 1])
        c = min(box_t[:, 2])
        if a < 0:
            ns0 = -math.ceil(-a)
        if b < 0:
            ns1 = -math.ceil(-b)
        if c < 0:
            ns2 = -math.ceil(-c)
        del a
        del b
        del c

        n0 = math.ceil(sum(abs(box_t[:, 0]))) + 1
        n1 = math.ceil(sum(abs(box_t[:, 1]))) + 1
        n2 = math.ceil(sum(abs(box_t[:, 2]))) + 1

        if not ns0 == ns1 == ns2 == 0:
            self.translate_atoms_f(np.array([ns0, ns1, ns2]))
        # extend atoms to wrap the MSL
        self.extend(n0, n1, n2)
        self.set_bvs(new_shape)
        self.truncate()
        if crit is not None:
            self.eliminate_overlap(crit)


    def transform_cell(self, t: np.ndarray):
        """
        transform the cell (the atoms will follow the deformation)

        :param t: 3x3 transform matrix (numpy array)
        :param extend_atoms: whether or not to add equivalent atoms
        :param crit: criteria for close atom elimination, None means no check
        """
        self.replace_bvs(np.array(np.matmul(t, self._bvs)))


    def transform(self, t: np.ndarray, crit: float=None):
        """
        transform both the cell and the atoms 
        NOTE: new equivalent atoms will be add to the lattice if the
              cell reaches outside of the original one, and outside atoms will
              be deleted.

        TODO: make this safer:
              what kind of transformation would not sabotage the original periodicity ?

        :param t: 3x3 transform matrix (numpy array)
        :param crit: criteria for close atom elimination, None means no check
        """
        check_2darray(t, 3, 3)
        # if t.dtype != int:
        #     raise ArgError("t", "must a 3x3 matrix of integers!")

        self.reshape(np.array(np.matmul(t, self._bvs)), crit=crit)


    def extend(self, nx: int, ny: int, nz: int):
        """
        build super cell

        :param nx: repeat times on x
        :param ny: repeat times on y
        :param nz: repeat times on z
        """
        if nx <= 0 or ny <= 0 or nz <=0:
            raise ArgError("nx, ny, nz", "must be positive integers!")

        t = np.array([\
                [nx, 0,  0],\
                [0, ny,  0],\
                [0,  0, nz]])

        # extend lattice vectors
        bv0 = self.get_bvs() # store a copy
        t = np.array([\
                [nx, 0,  0],\
                [0, ny,  0],\
                [0,  0, nz]])
        self._bvs = np.array(np.matmul(t, self._bvs))

        # directly extend atoms
        for e, atoms in self._apd.items():
            self._apd[e] = np.array(
                        [ a + i*bv0[0] + j*bv0[1] + k*bv0[2] 
                            for a in atoms 
                            for i in range(0, nx)
                            for j in range(0, ny)
                            for k in range(0, nz) ])

        self._cart2frac()


    def rotate_cell_around_axis(self, a: float, b: float, c: float):
        """
        rotate around Cartesian axis 

        :param a: angle on axis 0 (in degree)
        :param b: angle on axis 1 (in degree)
        :param c: angle on axis 2 (in degree)
        """ 
        self.replace_bvs(np.array(np.matmul(_calc_transform(a, b, c), self._bvs)))


    def rotate_atoms_around_axis(self, a: float, b: float, c: float):
        """
        rotate atoms around Cartesian axis 

        :param a: angle on axis 0 (in degree)
        :param b: angle on axis 1 (in degree)
        :param c: angle on axis 2 (in degree)
        """ 
        t = _calc_transform(a, b, c)
        for e, atoms in self._apd.items():
            self._apd[e] = np.array(np.matmul(atoms, t))
        self._cart2frac()


    # def rotate_around_bv(self, bv: int, angle: float):
    #     raise NotImplementedError("rotateAroundBV is not implemented yet!")


    def swap_bv(self, i: int, j: int):
        """
        swap two basis vector (i-th and j-th)

        :param i: the index of one bv
        :param j: the index of another bv
        """
        assert(0 <= i < 3)
        assert(0 <= j < 3)
        # exchange i-th and j-th rows of BVs
        self._bvs[[i, j]] = self._bvs[[j, i]]
        # exchange i-th and j-th columns of atom positions
        for e in self._apd.keys():
            self._apd[e][:, [i, j]] = self._apd[e][:, [j, i]]
            self._apd_f[e][:, [i, j]] = self._apd_f[e][:, [j, i]]


    def translate_atoms(self, tv: np.ndarray):
        """
        translate atom positions with a cartesian translation vector

        :param tv: translational vector (cartesian)
        """
        check_1darray(tv, 3)

        for e, atoms in self._apd.items():
            self._apd[e] = atoms + tv
        self._cart2frac()


    def translate_atoms_f(self, tv: np.ndarray):
        """
        translate atom positions with a fractional translation vector

        :param tv: translational vector
        """
        check_1darray(tv, 3)

        for e, atoms in self._apd_f.items():
            self._apd_f[e] = atoms + tv
        self._frac2cart()



###################################################################################################
# Definition of 2D Lattice class:
###################################################################################################

class Lattice2D(Lattice):
    """
    2D lattice class inherited from Lattice.

    Convention:
    1) The first and second basis vectors should be in the XY plane,
    and the 3rd bv direction is considered as the vaccum direction
    which should be on the Z direction.
    2) make sure the cell contains no such atom positions as
    (0, 0, 0.95) which is actually (0, 0, -0.05)

    NOTE: on initializing, vaccum space will be automatically added on the vertical direction
    """
    @staticmethod
    def check_bvs(bvs: np.ndarray):
        """ check if a basis vectors set meets the criterion for 2D lattice """
        if bvs[0][2] != 0 or bvs[1][2] != 0:
            raise ArgError("bvs", "the first 2 basis vectors should be in the XY plane!")
        elif bvs[2][2] == 0:
            raise ArgError("bvs", "the last basis vector is not on Z direction!")
        elif bvs[2][0] != 0 or bvs[2][1] != 0:
            raise ArgError("bvs", "the last lattice vector is not perpendicular to the x-y plane!")


    def __init__(self, sysname: str, bvs: np.ndarray, vaccum: float,
                 apd:Dict[str, np.ndarray]=None, apd_f:Dict[str, np.ndarray]=None):
        Lattice2D.check_bvs(bvs)

        super().__init__(sysname, bvs, apd=apd, apd_f=apd_f)
        self.set_vaccum(vaccum)
        # self.align_to_bottom()


    def set_c(self, c: float):
        "set the height of the 2D lattice"
        self._bvs[2][2] = float(c)
        super()._cart2frac()


    def get_c(self) -> float:
        return self._bvs[2][2]


    def get_vaccum(self):
        return self._vaccum


    def set_vaccum(self, vaccum: float):
        """
        this will set self._vaccum and update the basis vectors
        be sure to call this setter instead of changing self._vaccum directly!
        """
        self._vaccum = float(vaccum)
        self.set_c(self.thickness() + self._vaccum)
    

    def thickness(self) -> float:
        return self.max_z() - self.min_z()


    def align_to_bottom(self):
        """
        align atoms at Z=0
        NOTE: only call this if the cell contains no such atom positions as
              (0, 0, 0.95) which is actually (0, 0, -0.05)
        """
        super().translate_atoms(np.array([0, 0, -self.min_z()]))


    def max_z(self) -> float:
        "get the Z of the highest atom in cartesian coordinates"
        r = 0
        for atoms in self._apd.values():
            r = max(r, max(atoms[:, 2]))
        return r


    def min_z(self) -> float:
        "get the Z of the lowest atom in cartesian coordinates"
        r = math.inf
        for atoms in self._apd.values():
            r = min(r, min(atoms[:, 2]))
        return r


    def max_z_f(self) -> float:
        "get the Z in frac coordinates of the highest atom"
        r = 0
        for atoms in self._apd_f.values():
            r = max(r, max(atoms[:, 2]))
        return r


    def min_z_f(self) -> float:
        "get the Z of the lowest atom in frac coordinates "
        r = math.inf
        for atoms in self._apd_f.values():
            r = min(r, min(atoms[:, 2]))
        return r



###################################################################################################
# utility functions:
###################################################################################################


def lattice_to_2D(latt: Lattice, vaccum: float) -> Lattice2D:
    """build 2D lattice with a Lattice object"""
    return Lattice2D(latt.get_sysname(), latt.get_bvs(), vaccum, apd_f=latt.get_apd_f())


def merge_cell(latt0: Lattice, latt1: Lattice, sn='Merged System', tol=0.01) -> Lattice:
    """
    merge two lattices of close lattice shape and size

    :param latt0: lattice 0
    :param latt1: lattice 1
    :param sn: the name for the merged system
    :param tolerance: tolerance of lattice mismatch (fractional difference in BV components)

    :return: the merged lattice
    """
    # avoid operating on one same object
    if latt0 == latt1:
        raise ArgError('latt0, latt1', des='the two lattices must not be equivalent!')

    # avoid direct change on latt0 and latt1
    latt0 = copy.deepcopy(latt0)
    latt1 = copy.deepcopy(latt1)

    bv0 = latt0.get_bvs()
    bv1 = latt1.get_bvs()

    # check lattice mismatch
    diff = bv0 - bv1
    mismatch = max( [ abs(np.linalg.norm(diff[i]))/np.linalg.norm(bv0[i]) for i in range(3) ] )
    if mismatch > tol:
        raise Exception("the differences (%f) between the lattice lengths of latt0 and latt1 is too large, "
                        "try increase 'tolerance' (%f)" % (mismatch, tol))

    # merge fractional atom dict
    bv_m = bv1 + diff/2
    latt0.replace_bvs(bv0 - diff/2)
    latt1.replace_bvs(bv1 + diff/2)
    apd0 = latt0.get_apd_f()
    apd1 = latt1.get_apd_f()
    apd_m = {} # APD for merged system, fill it in:
    for e, ap in apd0.items():
        if e in apd1.keys():
            ap = np.row_stack((ap, apd1.pop(e)))
        apd_m[e] = ap

    for e, ap in apd1.items():
        apd_m[e] = ap

    return Lattice(sn, bv_m, apd_f=apd_m)


def read_lattice(fname: str, ftype="") -> Lattice:
    """
    read lattice from a file

    :param fname: file name
    :param ftype: file type (supported: 'vasp' and 'cif'), empty for auto-determination

    :return: read lattice
    """
    if ftype == "":
        ftype = __determine_filetype(fname)

    if ftype == "vasp":
        latt = read_poscar(fname)
    elif ftype == "cif":
        latt = read_cif(fname)
    else:
        raise UnSupportedError(ftype)

    return latt


def read_2D_lattice(fname: str, ftype="", vaccum=15) -> Lattice2D:
    """
    read 2D lattice from a file

    :param fname: file name
    :param ftype: file type (supported: 'vasp'), empty for auto-determination
    :param vaccum: vaccum on Z direction

    :return: read 2DLattice
    """
    return lattice_to_2D(read_lattice(fname, ftype), vaccum)


def __determine_filetype(fname: str) -> str:
    'Simple determination of file type from the suffix of the input files. (without check)'
    return os.path.basename(fname).split('.')[-1]


def read_xyz_as_lattice(fname: str, cell: np.ndarray, crit: float=0.1) -> Lattice:
    """
    read xyz file as lattice

    :param fname: file name
    :param cell: 3x3 ndarray basis vectors
    :param crit: overlap checking criteria (see Lattice.eliminate_overlap)

    :returns: read lattice
    """
    useful_lines = [] #type: List[str]
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line != "":
                useful_lines.append(line)

    apd = {}
    for i in range(2, int(useful_lines[0].strip())+2):
        items = useful_lines[i].strip().split()
        if items[0] not in apd.keys():
            apd[items[0]] = [ np.array( list( map(float, items[1:4]) )), ]
        else:
            apd[items[0]].append( np.array( list( map(float, items[1:4]) ) ) )
    
    for e in apd.keys():
        apd[e] = np.array(apd[e])

    latt = Lattice(useful_lines[1].strip(), cell, apd=apd)
    # latt.truncate()
    latt.eliminate_overlap(crit)

    return latt


def read_poscar(fname: str) -> str:
    """
    read a VASP POSCAR

    :param fname: file name

    :return: read lattice
    """
    # skip empty lines
    useful_lines = [] #type: List[str]
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line != "":
                useful_lines.append(line)

    # 1) system name line
    sysname = useful_lines[0]
    # 2) factor line
    factor = float(useful_lines[1])
    # 3-5) basis vectors lines, scale with `factor`
    bvs = factor * np.array([ list(map(float, useful_lines[i].split())) for i in (2, 3, 4) ])
    # 6) atom species line (omitting atom species line is not allowd here!!!)
    if useful_lines[5].isdigit():
        raise Exception("Omitting atom species line is not allowd!");

    elements = useful_lines[5].split()
    # 7) atom numbers line
    atom_nums = list(map(int, useful_lines[6].split())) # line 7 is atom numbers line

    # 8) coordinates type line (selective dynamics line will be SKIPPED here!!!)
    # skip selective dynamics line
    if useful_lines[7].capitalize().startswith('S'):
        n = 8
    else:
        n = 7

    # 9-EOF) atom position list
    coor_type = useful_lines[n].capitalize()[0]
    n += 1
    if coor_type in ("C", "K"): # Cartesian or Kartesian
        apd = {}
        for j, e in enumerate(elements):
            apd[e] = factor * np.array([ list(map(float, useful_lines[n+i].split())) for i in range(atom_nums[j]) ])
            n += atom_nums[j]
        return Lattice(sysname, bvs, apd=apd)
    elif coor_type in ("F", "D"): # Fractional or Direct
        apd_f = {}
        for j, e in enumerate(elements):
            apd_f[e] = np.array([ list(map(float, useful_lines[i+n].split())) for i in range(atom_nums[j]) ])
            n += atom_nums[j]
        return Lattice(sysname, bvs, apd_f=apd_f)
    else:
        raise UnSupportedError("Unkown coordinates type: %s specified in line 8!" % coor_type);


def read_cif(fname: str) -> Lattice:
    """
    read in a >>P 1 space group<< CIF (might be buggy since the format itself is like shit)

    NOTE: The _atom_site_xxx lines must be continous (with NO other commands between them),
          and the atom position lines must follow these _atom_site_xxx lines and be continous
          as well. Last but not least, it's best to leave a blank line after the end of the
          atom position lines block (so that we know the statement is ended)
    """
    # space_group = space_group_it_number = None
    name = a = b = c = alpha = beta = gamma = None
    i = 0
    header_read = False
    atom_pos_begin = False
    x_i = y_i = z_i = type_symbol_i = None
    apd_f = {} # type: Dict[str, List]

    with open(fname) as f:
        for line in f:
            sline = line.strip()
            items = sline.split()

            # skip comments and empty lines
            if line.startswith("#") or sline == "":
                continue

            # check if atom_pos beginned
            if atom_pos_begin:
                if header_read:
                    if len(items) != i: # this is NOT a reliable criteria
                        atom_pos_begin = False
                else:
                    if not sline.startswith("_atom_site"):
                        if None in (x_i, y_i, z_i, type_symbol_i):
                            raise Exception("Bad CIF: incomplete header !!!")
                        header_read = True
            elif sline.startswith("_atom_site"):
                atom_pos_begin = True

            if atom_pos_begin:
                if header_read:
                    x = float(items[x_i])
                    y = float(items[y_i])
                    z = float(items[z_i])
                    e = items[type_symbol_i]
                    if e not in apd_f.keys():
                        apd_f[e] = [np.array([x, y, z]), ]
                    else:
                        apd_f[e].append(np.array([x, y, z]))
                else:
                    if x_i is None and sline == "_atom_site_fract_x":
                        x_i = i
                    elif y_i is None and sline == "_atom_site_fract_y":
                        y_i = i
                    elif z_i is None and sline == "_atom_site_fract_z":
                        z_i = i
                    elif type_symbol_i is None and sline == "_atom_site_type_symbol":
                        type_symbol_i = i
                    i += 1
            elif name is None and sline.startswith("data_"):
                name = sline[5:]
            elif a is None and items[0] == "_cell_length_a":
                a = float(items[1])
            elif b is None and items[0] == "_cell_length_b":
                b = float(items[1])
            elif c is None and items[0] == "_cell_length_c":
                c = float(items[1])
            elif alpha is None and items[0] == "_cell_angle_alpha":
                alpha = float(items[1])
            elif beta is None and items[0] == "_cell_angle_beta":
                beta = float(items[1])
            elif gamma is None and items[0] == "_cell_angle_gamma":
                gamma = float(items[1])
            elif "space_group_name" in sline:
                # space_group = items[1]
                sg = "".join(items[1:])
                if sg not in ("'P1'", "P1"): # any other possiblities ?
                    raise Exception("I won't parse symmetry for you, only `P 1` space group is supported!\n")
                del sg
            elif "_space_group_IT_number" == items[0]:
                # space_group_it_number = items[1]
                if items[1] != "1":
                    raise Exception("I won't parse symmetry for you, only `P 1` space group is supported!\n")
            # else:
            #     print("skipped line:\n" + line)
    
    # list to ndarray
    for e in apd_f.keys():
        apd_f[e] = np.array(apd_f[e])

    # calc bvs (put bv_a on x, and bv_b on the x-y plane)
    fuck = np.array([c*math.cos(beta*math.pi/180) + c*math.cos(alpha*math.pi/180)*math.cos(gamma*math.pi/180),
                     c*math.cos(alpha*math.pi/180)*math.sin(gamma*math.pi/180)])
    bvs = np.array([ [a, 0, 0],
                     [b*math.cos(gamma*math.pi/180), b*math.sin(gamma*math.pi/180), 0],
                     [fuck[0], fuck[1], math.sin( math.acos(np.linalg.norm(fuck)/c) ) * c] ])
    
    return Lattice(name, bvs, apd_f=apd_f)


def save_xyz(latt: Lattice, fname: str):
    """
    write lattice to a xyz file in UNIX fileformat
    NOTE: fixed width output is used here,
          might cause problem if the numbers getting too large to fit in!

    :param latt: lattice to write
    :param fname: output file name
    """
    lines = []
    lines.append(str(latt.total_atom_num()) + "\n")
    lines.append(latt.get_sysname() + "\n")
    for e, atoms in latt.get_apd().items():
        for a in atoms:
            lines.append("%2s %11.7f %11.7f %11.7f\n" % (e, a[0], a[1], a[2]))

    with open(fname, 'w') as f:
        f.writelines(lines)


def save_poscar(latt: Lattice, fname: str, ctype='F'):
    """
    write lattice to a POSCAR file in UNIX fileformat
    NOTE: fixed width output is used here,
          might cause problem if the numbers getting too large to fit in!

    :param latt: lattice to write
    :param fname: output file name
    :param ctype: coordinates type: F (fractional) or C (cartesian)
    """
    content = ""
    # 1) sysname line
    content += latt.get_sysname() + '\n'
    # 2) factor line
    content += "   1.0\n"
    # 3-5) basis vectors line
    for v in latt.get_bvs():
        content += "   %20.16f  %20.16f  %20.16f\n" % tuple(v)

    # 8) coordinates type line
    line8 = ""
    apd_x = None # un-typed atom position dict
    if ctype == 'F':
        line8 += "Direct\n"
        apd_x = latt.get_apd_f()
    elif ctype == 'C':
        line8 += 'Cartesian\n'
        apd_x = latt.get_apd()
    else:
        raise UnSupportedError('coordinate type %s' % ctype)

    # 6) atom species line & 7) atom numbers line & 9) atom positions line
    andict = latt.atom_num_dict()
    line6 = "" # 6) atom species line
    line7 = "" # 7) atom numbers line
    line9 = "" # 9) atom position lines
    for i, (e, an) in enumerate(andict.items()):
        if i == 0:
            line6 += '   ' + e
            line7 += '    ' + str(an)
        else:
            line6 += '    ' + e
            line7 += '    ' + str(an)

        for ap in apd_x[e]:
            line9 += ' %20.16f %20.16f %20.16f\n' % tuple(ap)
    line6 += '\n'
    line7 += '\n'

    content += line6 + line7 + line8 + line9

    with open(fname, 'w') as f:
        f.write(content)


def save_cif(latt: Lattice, fname: str):
    lengths = latt.lattice_lengths()
    angles = latt.lattice_angles()
    content = """\
data_{sysname}

_chemical_name_common                  '{sysname}'
_cell_length_a                         {a}
_cell_length_b                         {b}
_cell_length_c                         {c}
_cell_angle_alpha                      {alpha}
_cell_angle_beta                       {beta}
_cell_angle_gamma                      {gamma}
_space_group_name_H-M_alt              'P 1'
_space_group_IT_number                 1

loop_
_space_group_symop_operation_xyz
   'x, y, z'

loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_U_iso_or_equiv
    _atom_site_adp_type
    _atom_site_occupancy\n""".format(sysname=latt.get_sysname(),
                                     a = lengths[0],
                                     b = lengths[1],
                                     c = lengths[2],
                                     alpha = angles[0],
                                     beta = angles[1],
                                     gamma = angles[2])
    
    for e, atoms in latt.get_apd_f().items():
        for i, p in enumerate(atoms):
            content += "{:<6s} {:<2s} {:10.6f} {:10.6f} {:10.6f} 0.00000  Uiso   1.00\n".format("%s%d" % (e, i+1), e, *p)

    with open(fname, "w") as f:
        f.write(content)




#############################################################################
# layeredstrc:
# functions for building common layered systems,
# including multi-layers and van der Waals heterostructures (vdWHs)
#############################################################################
def __calc_match(a: float, b: float, crit: float) -> Tuple[int]:
    """
    calculate the approximate integer (y, x) solution for `a*y = b*x`

    crit stands for the fractional maximum mismatch
    """
    n=1
    d = crit + 999999999999.0
    while d > crit:
        y = n
        x = round((a/b)*y)  # x/y ~= a/b; x * b ~= y * a
        # common routine to calc MCD (maximum common divisor)
        x0 = x
        y0 = y
        while x0 != y0:
            if x0 > y0:
                x0 = x0 - y0
            else:
                y0 = y0 - x0
        mcd = x0

        # simplify x : y
        x = int(x/mcd)
        y = int(y/mcd)
        d = abs(a * y - b * x) / min(a*y, b*x) # maximum mismatch ratio
        n += 1

    return y, x


def create_multilayers(latt: Lattice2D, ln: int, dist: float, vaccum: float=15,
                       sn: str=None) -> Lattice2D:
    """
    build multi-layer system from a monolayer
    NOTE: this will align the atoms to bottom first, make sure your input lattice
          doesn't contain atoms of different periods ((0.1, 0.1, 0.95) e. g.)

    :param latt: Lattice2D object as monolayer
    :param dist: interlayer distance
    :param vaccum: vaccum height
    :param ln: layer number
    :param sn: the new system name

    :returns: the built 2D lattice
    """
    if not isinstance(latt, Lattice2D):
        ArgError("latt")

    if sn is None:
        sysname = 'multilayer-' + latt.get_sysname()
    else:
        sysname = sn

    # make sure the vertical space is sufficient
    latt_i = copy.deepcopy(latt)
    latt_i.align_to_bottom() # align to bottom first
    t = latt_i.thickness()
    mc = t*ln + dist*(ln-1)
    if mc > latt_i.get_c():
        latt_i.set_c(mc)

    # merge lattices one by one accumulatively
    latt_m = copy.deepcopy(latt_i)
    for i in range(ln - 1):
        latt_i.translate_atoms(np.array([ 0, 0, (t + dist) * (i + 1) ]))
        latt_m = merge_cell(latt_m, latt_i, sn=sysname)

    latt_m = lattice_to_2D(latt_m, vaccum)
    return latt_m


def create_vdwh(latt0: Lattice2D, latt1: Lattice2D, dist: float, crit: float,
                vaccum: float=20, sn: str=None) -> Lattice2D:
    """
    build van der Waals Heterostructure (vdwh)
    NOTE: this will align the atoms to bottom first, make sure your input lattice
          doesn't contain atoms of different periods ((0.1, 0.1, 0.95) e. g.)

    :param latt0: lattice 0 (at bottom)
    :param latt1: lattice 1
    :param dist: interlayer distance (distance between closest atomic planes)
    :param crit: maximum fractional lattice length mismatch
    :param vaccum: vaccum height
    :param sn: the new system new
    """
    if not isinstance(latt0, Lattice2D):
        raise ArgError("latt0", "Lattice2D objects are expected!")
    if not isinstance(latt1, Lattice2D):
        raise ArgError("latt1", "Lattice2D objects are expected!")

    latt0_c = copy.deepcopy(latt0)
    latt1_c = copy.deepcopy(latt1)
    latt0_c.align_to_bottom()
    latt1_c.align_to_bottom()
    # make sure the vertical space is sufficient and identical
    safe_c = max( max(latt0_c.get_c(), latt1_c.get_c()),
                  latt0_c.max_z() + dist + latt1_c.max_z() )
    latt0_c.set_c(safe_c)
    latt1_c.set_c(safe_c)

    # making new system name
    if sn is None:
        sysname = latt0.get_sysname() + '/' + latt1.get_sysname() + ' vdWH'
    else:
        sysname = sn

    latt1_c.translate_atoms(np.array([0, 0, latt0_c.max_z() + dist]))

    # the lattice shape must be very close or identical
    if max(latt1_c.lattice_angles() - latt0_c.lattice_angles()) > 0.1:
        raise Exception("The shape of the two lattices differs too much!")

    lens0 = latt0_c.lattice_lengths()
    lens1 = latt1_c.lattice_lengths()

    # extend the two cells to match with each other if the diff in lattice length is large
    if max(abs(lens1 - lens0) / lens0) > crit:
        n0, n1 = __calc_match(lens0[0], lens1[0], crit)
        m0, m1 = __calc_match(lens0[1], lens1[1], crit)
        latt0_c.extend(n0, m0, 1)
        latt1_c.extend(n1, m1, 1)

    # 1.2 is for allowing trivial shape differences
    return lattice_to_2D(
        merge_cell(latt0_c, latt1_c, sn=sysname, tol=crit*1.2),
        vaccum=vaccum)



###################################################################################
# building 2D structures with Moire patterns:
###################################################################################
def calc_moire_bv(latt0: Lattice2D, latt1: Lattice2D) -> np.ndarray:
    """
    calculate the basis vectors of the Moirésuperlattice (refer to the AFM review)
    (Tang, K., Qi, W., Adv. Funct. Mater. 2020, 30, 2002672.
     https://doi.org/10.1002/adfm.202002672)

    :param latt0: lattice 0
    :param latt1: lattice 1

    :returns: the basis vectors of MSL calculated from the given two lattices
    """
    # refer to the AFM review article
    # (a is \alpha, w is \omega, D is \Delta)

    bv0 = latt0.get_bvs()
    bv1 = latt1.get_bvs()
    lens0 = latt0.lattice_lengths()
    lens1 = latt1.lattice_lengths()
    angles0 = latt0.lattice_angles()
    angles1 = latt1.lattice_angles()

    p1 = lens0[0]/lens1[0]
    p2 = lens0[1]/lens1[1]
    q = lens0[0]/lens0[1]
    w_1 = angles0[2] * math.pi/180
    w_2 = angles1[2] * math.pi/180
    w_ave = (w_1 + w_2) / 2
    cos_a1 = sum(bv0[0]*bv1[0])/(lens0[0] * lens1[0]);
    if cos_a1 > 1:
        cos_a1 = 1
    if cos_a1 < -1:
        cos_a1 = -1
    a1 = math.acos(cos_a1)
#    print(f"bv0[1] = {bv0[1]}, bv1[1] = {bv1[1]}; len[0][1] = {lens0[1]}, len[1][1] = {lens1[1]}")
#    neiji = sum(bv0[1]*bv1[1])
#    biaoji = lens0[1] * lens1[1]
#    print(f"neiji = {neiji}, biaoji = {biaoji}")
#    print(f"bv0[1] = {bv0[1]}, bv1[1] = {bv1[1]}; len[0][1] = {lens0[1]}, len[1][1] = {lens1[1]}")
    cos_a2 = sum(bv0[1] * bv1[1]) / (lens0[1] * lens1[1])
    if cos_a2 > 1:
        cos_a2 = 1
    if cos_a2 < -1:
        cos_a2 = -1
    a2 = math.acos(cos_a2)
    a_ave = (a1 + a2) / 2
    D = (math.sin(w_2) + p1*p2*math.sin(w_1) - math.sin(w_ave)*math.cos(a_ave)*(p1+p2) + math.cos(w_ave)*math.sin(a_ave)*(p1-p2)) / math.sin(w_1) 
    # P: 2x2
    P = (1/(D*math.sin(w_1))) * \
        np.array([\
            [ p1*(math.sin(w_1-a1)-p2*math.sin(w_1)),  q*p1*math.sin(a1)                       ],\
            [                     -p2*math.sin(a2)/q,  p2*(math.sin(w_1+a2)-p1*math.sin(w_1))  ] ])

    # extend P to 3x3
    P = [ [P[0][0], P[0][1], 0],\
          [P[1][0], P[1][1], 0],\
          [0,       0,       1] ]

    return np.array(np.matmul(np.array(P), np.matrix(bv0)), dtype=np.dtype(float))


def create_msl(latt0: Lattice2D, latt1: Lattice2D, alpha: float,
               dist: float, tol=1) -> Lattice2D:
    """
    Build Moire superlattice (MSL)

    :param latt0: lattice 0
    :param latt1: lattice 1
    :param alpha: twist angle
    :param dist: interlayer distance
    :param tol: maximum inter-atom distance (cartesian) to be considered as overlap

    :returns: the built MSL lattice
    """
    assert(isinstance(latt0, Lattice2D))
    assert(isinstance(latt1, Lattice2D))

    latt0_c = copy.deepcopy(latt0)
    latt1_c = copy.deepcopy(latt1)

    latt0_c.align_to_bottom()
    latt1_c.align_to_bottom()
    latt1_c.translate_atoms(np.array([0, 0, latt0_c.max_z() + dist]))

    latt1_c.rotate_cell_around_axis(0, 0, alpha)
    bv_m = calc_moire_bv(latt0_c, latt1_c)

    latt0_c.reshape(bv_m, tol)
    latt1_c.reshape(bv_m, tol)

    return merge_cell(latt0_c, latt1_c,
        sn=latt0_c.get_sysname() + '/' + latt1_c.get_sysname()+ str(alpha) + " deg rotated MSL")


def create_moire_sheet(latt0: Lattice2D, latt1: Lattice2D,
                       alpha: float, dist: float, w: float, len: float) -> Lattice2D:
    """
    build moire pattern 2D atomic sheets
    NOTE: for now, the atom sheet will be a rhombus shape, will vaccum space
          on each direction.

    :param latt0: lattice 0
    :param latt1: lattice 1
    :param alpha: rotation angle (degree)
    :param dist: interlayer distance 
    :param w: the angle of the rhombus shape (degree)
    :param len: the length of the rhombus shape

    :returns: the built lattice
    """
    w = w * math.pi / 180
    latt0_c = copy.deepcopy(latt0)
    latt1_c = copy.deepcopy(latt1)

    latt0_c.align_to_bottom()
    latt1_c.align_to_bottom()
    latt1_c.translate_atoms(np.array([0, 0, latt0_c.max_z() + dist]))

    latt1_c.rotate_atoms_around_axis(0, 0, alpha)

    v = 20
    bv = np.array([[len, 0, 0],
                   [len*math.cos(w), len*math.sin(w), 0],
                   [0, 0, v]])
    latt0_c.reshape(bv)
    latt1_c.reshape(bv)

    latt_m = merge_cell(latt0_c, latt1_c,
        sn=latt0_c.get_sysname() + "/" + latt1_c.get_sysname() + str(alpha) + "deg rotated Moire sheet")

    # add vaccum on each side
    len += v
    latt_m.set_bvs(np.array([[len,        0,           0],
                             [len*math.cos(w), len*math.sin(w), 0],
                             [0,           0,          20]]))

    return latt_m











#############################################################################
# tests
#############################################################################
class __TestData:
    def __init__(self):
        self.Graphene = """\
graphene                                
   1.00000000000000     
     2.4643431310023245    0.0000000000000000    0.0000000000000000
     1.2321715656837060    2.1341837548877978    0.0000000000000000
     0.0000000000000000    0.0000000000000000   15.0000000000000000
   C 
     2
Direct
 -0.0000000000000000 -0.0000000000000000  0.0000000000000000
  0.3333333340000024  0.3333333340000024  0.0000000000000000
 
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
"""

        self.MoS2 = """\
MoS2
1.0
        3.1600000858         0.0000000000         0.0000000000
        1.5800000429         2.7366403503         0.0000000000
        0.0000000000         0.0000000000        18.3299903870
   Mo    S
    1    2
Direct
     0.000000000         0.000000000         0.000000000
     0.333333334         0.333333334        -0.083666684
     0.333333334         0.333333334         0.083666684
"""


def __test_lattice():
    data = __TestData()
    if os.path.exists('MoS2.vasp'):
        os.remove('MoS2.vasp')
    with open('MoS2.vasp', 'w') as f:
        f.write(data.MoS2)

    # basic tests
    latt = read_lattice('MoS2.vasp')
    print(latt)
    save_poscar(latt, "out.vasp")
    if os.path.exists("out.vasp"):
        os.remove("out.vasp")

    save_xyz(latt, "MoS2.xyz")
    read_xyz_as_lattice("MoS2.xyz", latt.get_bvs())
    # latt_1 = read_xyz("out.xyz", latt.get_bvs())
    # print("latt_1:\n" + str(latt_1))
    # latt_2 = read_xyz("graphene.xyz", np.array([[2.46, 0, 0], [1.23, 2.13, 0], [0, 0, 15]]))
    # print("latt_2:\n" + str(latt_2))

    save_cif(latt, "MoS2.cif")
    latt_2 = read_cif("MoS2.cif")
    print("latt_2:\n" + str(latt_2))
    # latt_3 = read_cif("/home/tkw/Downloads/silicene.cif")
    # save_poscar(latt_3, "silicene.vasp")
    # print("latt_3:\n" + str(latt_3))

    assert(latt.get_sysname() == "MoS2")
    latt.set_sysname("new MoS2")
    assert(latt.get_sysname() == "new MoS2")
    latt.set_sysname("MoS2")
    bvs = np.array([[3.16, 0.0, 0.0],
                    [1.58, 2.74, 0.0],
                    [0.0, 0.0, 18.33]])
    assert(abs(latt.get_bvs() - bvs).max() < 0.1)
    del bvs
    assert(set(latt.atom_species()) == set(["Mo", "S"]))
    assert(latt.atom_num_dict() == {"Mo": 1, "S": 2})
    assert(np.linalg.norm(latt.lattice_lengths() - np.array([3.16, 3.16, 18.33])) < 0.1)
    assert(np.linalg.norm(latt.lattice_angles() - np.array([90, 90, 60])) < 0.1)
    assert(latt.total_atom_num() == 3)
    assert(latt.atom_num("S") == 2)
    # translate_atoms(_f) is not checked
    latt.add_atom_f("Mo", np.array([0.5, 0.5, 0.5]))
    assert(latt.atom_num("Mo") == 2)
    latt.delete_atom("Mo", 1)
    
    latt_tmp = copy.deepcopy(latt)
    latt_tmp.truncate()
    assert(latt_tmp.atom_num("S") == 1)
    assert(np.linalg.norm(latt_tmp.get_atom_position_f("S", 0) -\
                          np.array([0.33334, 0.33334, 0.08367])) < 0.01)

    latt_tmp = copy.deepcopy(latt)
    # print("--- testing Lattice.set_atom_position ---")
    latt_tmp.set_atom_position("Mo", 0, np.array([0.1, 0.1, 0.1]))
    assert(np.linalg.norm(
        latt_tmp.get_atom_position("Mo", 0) -\
        np.array([0.1, 0.1, 0.1])) < 0.0001)
    assert(np.linalg.norm(
        latt_tmp.get_atom_position_f("Mo", 0) -\
        np.array([0.01337499, 0.03654116, 0.00545554])) < 0.0001)
    # print("--- testing Lattice.set_atom_position_f ---")
    latt_tmp.set_atom_position_f("Mo", 0, np.array([0.1, 0.1, 0.1]))
    assert(np.linalg.norm(
        latt_tmp.get_atom_position("Mo", 0) -\
        np.array([0.47400001, 0.27366404, 1.83299904])) < 0.0001)
    assert(np.linalg.norm(
        latt_tmp.get_atom_position_f("Mo", 0) -\
        np.array([0.1, 0.1, 0.1])) < 0.0001)
    assert(abs(latt.atom_distance("Mo", 0, "S", 1) - 2.38) < 0.1)


    print("--- testing Lattice.set_bvs ---")
    latt_tmp = copy.deepcopy(latt)
    bvs_new = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 20]])
    print("set bvs as:\n" + str(bvs_new))
    latt_tmp.set_bvs(bvs_new)
    assert(abs(latt_tmp.get_bvs() - bvs_new).max() < 0.000001)
    print("saving ...")
    save_poscar(latt_tmp, "set_bvs.vasp")
    del bvs_new

    print("--- testing Lattice.replace_bvs ---")
    latt_tmp = copy.deepcopy(latt)
    bvs_new = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 20]])
    latt_tmp.replace_bvs(bvs_new)
    print("replace bvs as:\n" + str(bvs_new))
    print("saving ...")
    save_poscar(latt_tmp, "replace_bvs.vasp")

    print("--- testing Lattice.extend ---")
    print("extend to 2x2x1 ...")
    latt_tmp = copy.deepcopy(latt)
    latt_tmp.extend(3, 3, 1)
    # print(latt_tmp)
    save_poscar(latt_tmp, "extend.vasp")
    print("--- testing Lattice.rotate_atoms_around_axis ---")
    print("rotate around Z for 10 degree ...")
    latt_tmp.rotate_atoms_around_axis(0, 0, 10)
    # print(latt_tmp)
    print("saving ...")
    save_poscar(latt_tmp, "rotate_atoms_around_axis.vasp")

    print("--- testing Lattice.rotate_cell_around_axis ---")
    print("rotate around Z for 180 degree ...")
    latt_tmp = copy.deepcopy(latt)
    latt_tmp.rotate_cell_around_axis(0, 0, 180)
    print("saving ...")
    save_poscar(latt_tmp, "rotate_cell_around_axis.vasp")

    print("--- testing Lattice.swap_bv ---")
    print("swap 0-th and 2-th bv")
    latt_tmp = copy.deepcopy(latt)
    latt_tmp.swap_bv(0, 2)
    print(latt_tmp)

    print("--- testing Lattice.confine_atoms_to_lattice ---")
    latt_tmp = copy.deepcopy(latt)
    latt_tmp.confine_atoms_to_lattice()
    print(latt_tmp)

    print("--- testing Lattice.reshape ---")
    latt_tmp = copy.deepcopy(latt)
    latt_tmp.reshape( np.array([[50, 0, 0], [0, 50, 0], latt_tmp.get_bvs()[2]]) )
    print("saving ...")
    save_poscar(latt_tmp, "reshape.vasp")

    print("--- testing Lattice.transform ---")
    latt_tmp = copy.deepcopy(latt)
    # latt_tmp.transform( np.array([[1, 0, 0], [-1, 1, 0], [0, 0, 1]]) )
    latt_tmp.transform( np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 1]]) )
    # latt_tmp.transform( np.array([[-1, 0, 0], [1, 1, 0], [0, 0, 1]]) )
    # latt_tmp.transform( np.array([[-2, 0, 0], [2, 1, 0], [0, 0, 1]]) )
    # latt_tmp.transform( np.array([[2, 0, 0], [2, 1, 0], [0, 0, 1]]) )
    print("saving ...")
    save_poscar(latt_tmp, "transform.vasp")

    print("--- testing Lattice2D.align_to_bottom ---")
    latt_tmp = lattice_to_2D(latt, 20)
    latt_tmp.align_to_bottom()
    print(latt_tmp)


def __test_lattice2d():
    data = __TestData()
    if os.path.exists('MoS2.vasp'):
        os.remove('MoS2.vasp')
    with open('MoS2.vasp', 'w') as f:
        f.write(data.MoS2)

    latt = read_2D_lattice("MoS2.vasp")
    latt.translate_atoms_f(np.array([0, 0, 0.5]))
    save_poscar(latt, "x.vasp")
    latt.align_to_bottom()
    save_poscar(latt, "y.vasp")


def __test_layeredstrc():
    data = __TestData()
    if os.path.exists('MoS2.vasp'):
        os.remove('MoS2.vasp')
    with open('MoS2.vasp', 'w') as f:
        f.write(data.MoS2)

    if os.path.exists('Graphene.vasp'):
        os.remove('Graphene.vasp')
    with open('Graphene.vasp', 'w') as f:
        f.write(data.Graphene)

    latt0 = read_2D_lattice('Graphene.vasp')
    latt1 = read_2D_lattice('MoS2.vasp')
    dist = 3.0
    crit = 0.05
    save_poscar(create_vdwh(latt0, latt1, dist, crit), 'vdwh.vasp')
    save_poscar(create_multilayers(latt1, 2, dist), "bilayer.vasp")


def __test_moire():
    data = __TestData()
    if os.path.exists('MoS2.vasp'):
        os.remove('MoS2.vasp')
    with open('MoS2.vasp', 'w') as f:
        f.write(data.MoS2)
    if os.path.exists('Graphene.vasp'):
        os.remove('Graphene.vasp')
    with open('Graphene.vasp', 'w') as f:
        f.write(data.Graphene)

    latt0 = read_2D_lattice("MoS2.vasp")
    latt1 = read_2D_lattice("Graphene.vasp")
    print("--- testing create_msl ---")
    latt_msl = create_msl(latt0, latt1, 3, 3)
    latt_ms = create_moire_sheet(latt0, latt1, 3, 3, 90, 100)
    print("saving ...")
    save_poscar(latt_msl, "msl.vasp")
    save_poscar(latt_ms, "ms.vasp")


# if __name__ == '__main__':
#    __test_lattice()
    # __test_lattice2d()
    # __test_layeredstrc()
    # __test_moire()
