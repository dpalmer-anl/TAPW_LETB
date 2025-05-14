import numpy as np
import scipy.linalg as sla
from scipy import sparse
import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_gk as mgk
import mtbmtbg.moire_io as mio
import mtbmtbg.moire_tb as mtb
import mtbmtbg.moire_analysis as mta
import mtbmtbg.moire_letb as letb
import mtbmtbg.moire_cont as mc 
import time
from mtbmtbg.config import TBInfo, DataType, EngineType, ValleyType
import matplotlib.pyplot as plt

VPI_0 = TBInfo.VPI_0
VSIGMA_0 = TBInfo.VSIGMA_0
R_RANGE = TBInfo.R_RANGE

TB_models = {"MK":mtb._sk_integral,
            "LETB":None}

def k_path(sym_pts,nk,report=False):
        r"""
    
        Interpolates a path in reciprocal space between specified
        k-points.  In 2D or 3D the k-path can consist of several
        straight segments connecting high-symmetry points ("nodes"),
        and the results can be used to plot the bands along this path.
        
        The interpolated path that is returned contains as
        equidistant k-points as possible.
    
        :param kpts: Array of k-vectors in reciprocal space between
          which interpolated path should be constructed. These
          k-vectors must be given in reduced coordinates.  As a
          special case, in 1D k-space kpts may be a string:
    
          * *"full"*  -- Implies  *[ 0.0, 0.5, 1.0]*  (full BZ)
          * *"fullc"* -- Implies  *[-0.5, 0.0, 0.5]*  (full BZ, centered)
          * *"half"*  -- Implies  *[ 0.0, 0.5]*  (half BZ)
    
        :param nk: Total number of k-points to be used in making the plot.
        
        :param report: Optional parameter specifying whether printout
          is desired (default is True).

        :returns:

          * **k_vec** -- Array of (nearly) equidistant interpolated
            k-points. The distance between the points is calculated in
            the Cartesian frame, however coordinates themselves are
            given in dimensionless reduced coordinates!  This is done
            so that this array can be directly passed to function
            :func:`pythtb.tb_model.solve_all`.

          * **k_dist** -- Array giving accumulated k-distance to each
            k-point in the path.  Unlike array *k_vec* this one has
            dimensions! (Units are defined here so that for an
            one-dimensional crystal with lattice constant equal to for
            example *10* the length of the Brillouin zone would equal
            *1/10=0.1*.  In other words factors of :math:`2\pi` are
            absorbed into *k*.) This array can be used to plot path in
            the k-space so that the distances between the k-points in
            the plot are exact.

          * **k_node** -- Array giving accumulated k-distance to each
            node on the path in Cartesian coordinates.  This array is
            typically used to plot nodes (typically special points) on
            the path in k-space.
        """
    
        k_list=np.array(sym_pts)
    
        # number of nodes
        n_nodes=k_list.shape[0]
    
        mesh_step = nk//(n_nodes-1)
        mesh = np.linspace(0,1,mesh_step)
        step = (np.arange(0,mesh_step,1)/mesh_step)
    
        kvec = np.zeros((0,3))
        knode = np.zeros(n_nodes)
        for i in range(n_nodes-1):
           n1 = k_list[i,:]
           n2 = k_list[i+1,:]
           diffq = np.outer((n2 - n1),  step).T + n1
    
           dn = np.linalg.norm(n2-n1)
           knode[i+1] = dn + knode[i]
           if i==0:
              kvec = np.vstack((kvec,diffq))
           else:
              kvec = np.vstack((kvec,diffq))
        kvec = np.vstack((kvec,k_list[-1,:]))
    
        dk_ = np.zeros(np.shape(kvec)[0])
        for i in range(1,np.shape(kvec)[0]):
           dk_[i] = np.linalg.norm(kvec[i,:]-kvec[i-1,:]) + dk_[i-1]
    
        return (kvec,dk_, knode)


def _set_const_mtrx(
        n_moire: int,
        npair_dict: dict,
        ndist_dict: dict,
        m_basis_vecs: dict,
        g_vec_list: np.ndarray,
        atom_pstn_list: np.ndarray,
        model: str
) -> dict:
    """setup constant matrix in calculating TBPLW

    Args:
        n_moire (int): an integer describing the moire system
        npair_dict (dict): neighbour pair dictionary
        ndist_dict (dict): neighbour distance dictionary
        m_basis_vecs (dict): moire basis vectors dictionary
        g_vec_list (np.ndarray): Glist (Attention! should be sampled near specific `VALLEY`)
        atom_pstn_list (np.ndarray): atom postions in a moire unit cell

    Raises:
        Exception: Hopping matrix is not Hermitian
        Exception: Overlap matrix is not Hermitian

    Returns:
       dict: {gr_mtrx, tr_mtrx, sr_mtrx}
    """

    # read values
    row, col = npair_dict['r'], npair_dict['c']
    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    n_g = g_vec_list.shape[0]
    n_atom = atom_pstn_list.shape[0]
    # normalize factor
    factor = 1/np.sqrt(n_atom/4)

    gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g, r[:2])) for g in g_vec_list for r in atom_pstn_list
                       ]).reshape(n_g, n_atom)

    g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
    gr_mtrx = sla.block_diag(g1, g2, g3, g4)

    
    if model=="MK":
        hopping = mtb._sk_integral(ndist_dict)
    elif model=="LETB":
        lattice_vectors = np.zeros((3,3))
        lattice_vectors[0,:2] = m_basis_vecs['mu1']
        lattice_vectors[1,:2] = m_basis_vecs['mu2']
        hopping = letb.letb(atom_pstn_list,lattice_vectors,npair_dict)

    tr_mtrx = sparse.csr_matrix((hopping, (row, col)), shape=(n_atom, n_atom))
    tr_mtrx = (tr_mtrx + (tr_mtrx.transpose()).conjugate())/2
    tr_mtrx_cc = (tr_mtrx.transpose()).conjugate()
    tr_mtrx_delta = tr_mtrx-tr_mtrx_cc

    if tr_mtrx_delta.max()>1.0e-9:
        print(tr_mtrx_delta.max())
        raise Exception("Hopping matrix is not hermitian?!")

    diag_ones = sparse.diags([1 for i in range(n_atom)])
    sr_mtrx = gr_mtrx@(diag_ones@(gr_mtrx.conj().T))
    sr_mtrx_cc = (sr_mtrx.transpose()).conjugate()
    sr_mtrx_delta = sr_mtrx-sr_mtrx_cc

    if sr_mtrx_delta.max()>1.0e-9:
        print(sr_mtrx_delta.max())
        raise Exception("Overlap matrix is not hermitian?!")

    const_mtrx_dict = {}
    const_mtrx_dict['gr'] = gr_mtrx
    const_mtrx_dict['tr'] = tr_mtrx
    const_mtrx_dict['sr'] = sr_mtrx

    return const_mtrx_dict


def TB_solve(model,
              n_moire: int,
              n_g: int,
              n_k: int,
              disp: bool = True,
              datatype=DataType.CORRU,
              engine=EngineType.TBPLW,
              valley=ValleyType.VALLEYK1,
              cutoff=5.29) -> dict:
    """tight binding solver for TBG

    Args:
        n_moire (int): an integer describing the size of commensurate TBG systems
        n_g (int): Glist size, n_g = 5 for MATBG
        n_k (int): n_k 
        disp (bool): whether calculate dispersion
        datatype (DataType, optional): atom data type. Defaults to DataType.CORRU.
        engine (EngineType, optional): TB solver engine type. Defaults to EngineType.TBPLW.
        valley (EngineType, optional): valley concerned. Defaults to EngineType.VALLEYK1.

    Returns:
        dict:         
        'emesh': np.array(emesh),
        'dmesh': np.array(dmesh),
        'kline': kline,
        'trans': transmat_list,
        'nbmap': neighbor_map
    """
    start_time = time.process_time()
    dmesh = []
    emesh = []
    kline = 0
    emax = -1000
    emin = 1000
    count = 1

    # load atom data
    atom_pstn_list = mio.read_atom_pstn_list(n_moire, datatype)
    # construct moire info
    (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
    (all_nns, enlarge_atom_pstn_list) = mset.set_atom_neighbour_list(atom_pstn_list, m_basis_vecs,distance=cutoff)
    (npair_dict, ndist_dict) = mset.set_relative_dis_ndarray(atom_pstn_list, enlarge_atom_pstn_list, all_nns)
    # set up g list
    o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
    # move to specific valley or combined valley
    g_vec_list = mtb._set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, valley)
    # constant matrix dictionary
    const_mtrx_dict = _set_const_mtrx(n_moire, npair_dict, ndist_dict, m_basis_vecs, g_vec_list, atom_pstn_list,model)
    # constant list
    (transmat_list, neighbor_map) = mgk.set_kmesh_neighbour(n_g, m_basis_vecs, o_g_vec_list)

    if disp:
        (kline, kmesh) = mgk.set_tb_disp_kmesh(n_k, high_symm_pnts)
    else:
        kmesh = mgk.set_kmesh(n_k, m_basis_vecs)

    n_atom = atom_pstn_list.shape[0]
    n_band = g_vec_list.shape[0]*4
    n_kpts = kmesh.shape[0]
    print("="*100)
    print("num of atoms".ljust(30), ":", n_atom)
    print("num of kpoints".ljust(30), ":", n_kpts)
    print("num of bands".ljust(30), ":", n_band)
    print("="*100)
    setup_time = time.process_time()

    for k_vec in kmesh:
        print("k sampling process, counter:", count)
        count += 1
        hamk = mtb._cal_hamiltonian_k(ndist_dict, npair_dict, const_mtrx_dict, k_vec, n_atom, engine)
        eigen_val, eigen_vec = mtb._cal_eigen_hamk(hamk, const_mtrx_dict['sr'], datatype, engine)
        if np.max(eigen_val)>emax:
            emax = np.max(eigen_val)
        if np.min(eigen_val)<emin:
            emin = np.min(eigen_val)
        emesh.append(eigen_val)
        dmesh.append(eigen_vec)
    comp_time = time.process_time()

    print("="*100)
    print("emax =", emax, "emin =", emin)
    print("="*100)
    print("set up time:", setup_time-start_time, "comp time:", comp_time-setup_time)
    print("="*100)

    return {
        'emesh': np.array(emesh),
        'dmesh': np.array(dmesh),
        'kline': kline,
        'trans': transmat_list,
        'nbmap': neighbor_map
    }

def get_BM_model(model,
              n_moire: int,
              n_g: int,
              n_k: int,
              disp: bool = True,
              datatype=DataType.CORRU,
              engine=EngineType.TBPLW,
              valley=ValleyType.VALLEYK1,
              cutoff=5.29) -> dict:
    """tight binding solver for TBG

    Args:
        n_moire (int): an integer describing the size of commensurate TBG systems
        n_g (int): Glist size, n_g = 5 for MATBG
        n_k (int): n_k 
        disp (bool): whether calculate dispersion
        datatype (DataType, optional): atom data type. Defaults to DataType.CORRU.
        engine (EngineType, optional): TB solver engine type. Defaults to EngineType.TBPLW.
        valley (EngineType, optional): valley concerned. Defaults to EngineType.VALLEYK1.

    Returns:
        dict:         
        'emesh': np.array(emesh),
        'dmesh': np.array(dmesh),
        'kline': kline,
        'trans': transmat_list,
        'nbmap': neighbor_map
    """
    start_time = time.process_time()
    dmesh = []
    emesh = []
    kline = 0
    emax = -1000
    emin = 1000
    count = 1

    # load atom data
    atom_pstn_list = mio.read_atom_pstn_list(n_moire, datatype)
    # construct moire info
    (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
    (all_nns, enlarge_atom_pstn_list) = mset.set_atom_neighbour_list(atom_pstn_list, m_basis_vecs,distance=cutoff)
    (npair_dict, ndist_dict) = mset.set_relative_dis_ndarray(atom_pstn_list, enlarge_atom_pstn_list, all_nns)
    # set up g list
    o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
    # move to specific valley or combined valley
    g_vec_list = mtb._set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, valley)
    # constant matrix dictionary
    const_mtrx_dict = _set_const_mtrx(n_moire, npair_dict, ndist_dict, m_basis_vecs, g_vec_list, atom_pstn_list,model)

    # constant list
    (transmat_list, neighbor_map) = mgk.set_kmesh_neighbour(n_g, m_basis_vecs, o_g_vec_list)

    if disp:
        (kline, kmesh) = mgk.set_tb_disp_kmesh(n_k, high_symm_pnts)
    else:
        kmesh = mgk.set_kmesh(n_k, m_basis_vecs)

    n_atom = atom_pstn_list.shape[0]
    n_band = g_vec_list.shape[0]*4
    n_kpts = kmesh.shape[0]
    print("="*100)
    print("num of atoms".ljust(30), ":", n_atom)
    print("num of kpoints".ljust(30), ":", n_kpts)
    print("num of bands".ljust(30), ":", n_band)
    print("="*100)
    setup_time = time.process_time()

    full_ham = []
    for k_vec in kmesh:
        print("k sampling process, counter:", count)
        count += 1
        hamk =mtb. _cal_hamiltonian_k(ndist_dict, npair_dict, const_mtrx_dict, k_vec, n_atom, engine)
        full_ham.append(hamk)
    comp_time = time.process_time()

    print("="*100)
    print("set up time:", setup_time-start_time, "comp time:", comp_time-setup_time)
    print("="*100)

    return np.array(full_ham)

def get_cont_model(n_moire: int, n_g: int, n_k: int, disp: bool = True, valley: int = 1,solve=False) -> dict:
    """
    continuum model solver for TBG system
    """

    dmesh = []
    emesh = []
    kline = 0
    emax = -1000
    emin = 1000
    count = 1
    # construct moire info
    rt_angle_r, rt_angle_d = mset._set_moire_angle(n_moire)
    rt_mtrx_half = mset._set_rt_mtrx(rt_angle_r/2)
    (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
    # set up g list
    o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
    # move to specific valley or combined valley
    g_vec_list = mc._set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, valley)
    
    # interlayer interaction
    tmat = mc._make_t(g_vec_list, m_basis_vecs, valley)
    # atomic K points
    kpts = mc._set_kpt(rt_mtrx_half)

    if disp:
        (kline, kmesh) = mgk.set_tb_disp_kmesh(n_k, high_symm_pnts)
    else:
        kmesh = mgk.set_kmesh(n_k, m_basis_vecs)
    full_ham = []
    for k in kmesh:
        print("k sampling process, counter:", count)
        count += 1
        hamk = mc._make_hamk(k, kpts, g_vec_list, rt_mtrx_half, tmat, valley)
        if solve:
            eigen_val, eigen_vec = np.linalg.eigh(hamk)
            emesh.append(eigen_val)
            dmesh.append(eigen_vec)
        else:
            full_ham.append(hamk)
    if solve:
        return {'emesh': np.array(emesh), 'dmesh': np.array(dmesh), 'kline': kline}
    else:
        return np.array(full_ham)

def plot_bands(evals,kline,erange=0.1,title="",figname="bands.png"):
    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel("Path in k-space")
    
    nbands = np.shape(evals)[0]
    efermi = np.mean([evals[nbands//2,0],evals[(nbands-1)//2,0]])
    fermi_ind = (nbands)//2

    for n in range(np.shape(evals)[0]):
        ax.plot(kline,evals[n,:]-efermi,color="black")
        
    # make an PDF figure of a plot
    #fig.tight_layout()
    ax.set_ylim(-erange,erange)
    ax.set_ylabel(r'$E - E_F$ (eV)')
    fig.savefig(figname, bbox_inches='tight')
    plt.clf()

def get_moire_potential(bm_ham):
    basis_dim = np.shape(bm_ham)[1]
    U = bm_ham[basis_dim//2:,:basis_dim//2].real
    udim = np.shape(U)[0]
    U_a1a2 = U[:udim//2,:udim//2]
    U_a1b2 = U[udim//2:,:udim//2]
    U_b1a2 = U[:udim//2,udim//2:]
    U_b1b2 = U[udim//2:,udim//2:]

    w_a1a2_G = np.zeros((2,2,udim//4))
    w_a1b2_G = np.zeros((2,2,udim//4))
    w_b1a2_G = np.zeros((2,2,udim//4))
    w_b1b2_G = np.zeros((2,2,udim//4))

    for i in range(udim//4):
        w_a1a2_G[:,:,i] = U_a1a2[2*i:2*i+2, 2*i:2*i+2]
        w_a1b2_G[:,:,i] = U_a1b2[2*i:2*i+2, 2*i:2*i+2]
        w_b1a2_G[:,:,i] = U_b1a2[2*i:2*i+2, 2*i:2*i+2]
        w_b1b2_G[:,:,i] = U_b1b2[2*i:2*i+2, 2*i:2*i+2]
    return np.array(w_a1a2_G), np.array(w_a1b2_G), np.array(w_b1a2_G), np.array(w_b1b2_G)

def get_moire_transfer_mat(bm_ham,glist,m_basis_vecs,valley=1,basis="continuum"):

    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    # three nearest g vec
    g1 = np.array([0, 0])
    g2 = -valley*m_g_unitvec_2
    g3 = valley*m_g_unitvec_1
    omega1, omega2 = np.exp(1j*2*np.pi/3)**valley, np.exp(-1j*2*np.pi/3)**valley

    basis_dim = np.shape(bm_ham)[1]
    glist_size = np.shape(glist)[0]
    tmat = bm_ham[basis_dim//2:,:basis_dim//2]

    T1 = []
    T2 = []
    T3 = []

    g_T1 = []
    g_T2 = []
    g_T3 = []
    
    if basis=="continuum":
        for i in range(glist_size//2):
            for j in range(glist_size//2):
                #for basis of continuum model i.e. [g1[[a1a2,a1b2],[a2b1,b1b2]], g2[[]]] 
                delta_k = glist[i]-glist[j]
                # matrix element in three cases:
                if mc._check_eq(delta_k, g1):
                    T1.append(list(tmat[2*i:2*i+2, 2*j:2*j+2]))
                    g_T1.append(list(glist[i]))
                if mc._check_eq(delta_k, g2):
                    t2 = tmat[2*i:2*i+2, 2*j:2*j+2]
                    t2[0,1] /= omega2
                    t2[1,0] /= omega1
                    T2.append(list(t2))
                    g_T2.append(list(glist[i]))
                if mc._check_eq(delta_k, g3):
                    t3 = tmat[2*i:2*i+2, 2*j:2*j+2]
                    t3[0,1] /= omega1
                    t3[1,0] /= omega2
                    T3.append(list(t3))
                    g_T3.append(list(glist[i]))

    elif basis=="TAPW":
        for i in range(glist_size):
            for j in range(glist_size):
                #for basis of continuum model i.e. [a1a2[g1,g2...],a1b2[g1,g2,...],a2b1[g1,g2,...],b1b2[g1,g2,...]]
                delta_k = glist[i]-glist[j]
                # matrix element in three cases:
                ua1a2 = tmat[i,j]
                ua1b2 = tmat[i+glist_size,j]
                ua2b1 = tmat[i,j+glist_size]
                ub1b2 = tmat[i+glist_size,j+glist_size]
                t = np.array([[ua1a2,ua1b2],[ua2b1,ub1b2]])
                if mc._check_eq(delta_k, g1):
                    T1.append(list(t))
                    g_T1.append(list(glist[i]))
                if mc._check_eq(delta_k, g2):
                    t[0,1] /= omega2
                    t[1,0] /= omega1
                    T2.append(list(t))
                    g_T2.append(list(glist[i]))
                if mc._check_eq(delta_k, g3):
                    t[0,1] /= omega1
                    t[1,0] /= omega2
                    T3.append(list(t))
                    g_T3.append(list(glist[i]))
    return np.array(T1), np.array(T2), np.array(T3), np.array(g_T1), np.array(g_T2), np.array(g_T3)

if __name__=="__main__":
    n_moire = [4] #16,20,30,33,35]
    n_moire = [30] #30,32,34,36,48
    n_g= 5
    n_k = 10
    engine=EngineType.TBPLW
    disp=False
    valley=ValleyType.VALLEYK1
    valleyint = 1

    models = ["LETB"]
    """for n in n_moire:
        (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n)
        # set up g list
        o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
        g_vec_list = mc._set_g_vec_list_valley(n, o_g_vec_list, m_basis_vecs, valleyint)
        
        for m in models:
            theta= np.arcsin(np.sqrt(3)*(2*n+1)/(6*n**2 + 6*n + 2))*180/np.pi
            print(100*"=")
            print("n moire = ",n)
            print("Theta = ",theta)
            
            #continuum model
            bm_ham = get_cont_model(n, n_g, n_k, disp=disp) #, valley=valley)
            
            T1, T2, T3, g_T1, g_T2, g_T3 = get_moire_transfer_mat(bm_ham[0,:,:],g_vec_list,m_basis_vecs,valley=valleyint,basis="continuum")
            print("BM model")
            print("<T1> = ",np.mean(T1,axis=0).real)
            print("<T2> = ",np.mean(T2,axis=0).real)
            print("<T3> = ",np.mean(T3,axis=0).real)
            
            np.savez("BM_Ham/BM_Ham_continuum_relaxed_theta_"+str(theta),bm_ham = bm_ham,T1 = T1, T2 = T2, T3=T3)
            ham_dim=np.shape(bm_ham)[1]
            U = bm_ham[0,:ham_dim//2,ham_dim//2:]
            plt.imshow(U.real)
            plt.title("BM, "+r"$\theta=$"+str(theta))
            plt.colorbar()
            plt.savefig("figures/BM_Ham_continuum_relaxed_theta_"+str(theta)+".png")
            plt.clf()
            
            # unrelaxed
            bm_ham = get_BM_model(m,n, n_g, n_k, datatype=DataType.RIGID,disp=disp, engine=engine, valley=valleyint)
            T1, T2, T3, g_T1, g_T2, g_T3 = get_moire_transfer_mat(bm_ham[0,:,:],g_vec_list,m_basis_vecs,valley=1,basis="TAPW")
            print("Unrelaxed")
            print("<T1> = ",np.mean(T1,axis=0).real)
            print("<T2> = ",np.mean(T2,axis=0).real)
            print("<T3> = ",np.mean(T3,axis=0).real)

            np.savez("BM_Ham/BM_Ham_"+m+"_unrelaxed_theta_"+str(theta),bm_ham = bm_ham,T1 = T1, T2 = T2, T3=T3)
            ham_dim=np.shape(bm_ham)[1]
            U = bm_ham[0,:ham_dim//2,ham_dim//2:]
            plt.imshow(U.real)
            plt.title("Unrelaxed, "+r"$\theta=$"+str(theta))
            plt.colorbar()
            plt.savefig("figures/BM_Ham_"+m+"_unrelaxed_theta_"+str(theta)+".png")
            plt.clf()
            
            #relaxed
            bm_ham = get_BM_model(m,n, n_g, n_k, datatype=DataType.RELAX, disp=disp, engine=engine, valley=valley)
            T1, T2, T3, g_T1, g_T2, g_T3 = get_moire_transfer_mat(bm_ham[0,:,:],g_vec_list,m_basis_vecs,valley=1,basis="TAPW")
            print("Relaxed")
            print("<T1> = ",np.mean(T1,axis=0).real)
            print("<T2> = ",np.mean(T2,axis=0).real)
            print("<T3> = ",np.mean(T3,axis=0).real)
            np.savez("BM_Ham/BM_Ham_"+m+"_relaxed_theta_"+str(theta),bm_ham = bm_ham,T1 = T1, T2 = T2, T3=T3)
            ham_dim=np.shape(bm_ham)[1]
            U = bm_ham[0,:ham_dim//2,ham_dim//2:]
            plt.imshow(U.real)
            plt.colorbar()
            plt.title("Relaxed, "+r"$\theta=$"+str(theta))
            plt.savefig("figures/BM_Ham_"+m+"_relaxed_theta_"+str(theta)+".png")
            plt.clf()"""

    cutoff = 5.29
    for n in n_moire:
        for m in models:
            theta= np.arcsin(np.sqrt(3)*(2*n+1)/(6*n**2 + 6*n + 2))*180/np.pi
            results = TB_solve(m,n, n_g, 50, datatype=DataType.RIGID,disp=True, engine=engine, valley=valley,cutoff=cutoff)
            evals = results["emesh"].T
            kline = results["kline"]

            plot_bands(evals,kline,erange=0.1,title="unrelaxed "+r"$\theta=$"+str(theta),figname = "figures/BM_Ham_"+m+"_unrelaxed_theta_"+str(theta)+"_bands.png")

            """results = TB_solve(m,n, n_g, 50, datatype=DataType.RELAX,disp=True, engine=engine, valley=valley)
            evals = results["emesh"].T
            kline = results["kline"]
            plot_bands(evals,kline,title="relaxed "+r"$\theta=$"+str(theta),figname = "figures/BM_Ham_"+m+"_relaxed_theta_"+str(theta)+"_bands.png")"""
