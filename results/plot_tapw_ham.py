import numpy as np
from write_BM_model import *
import matplotlib.pyplot as plt
import glob
if __name__=="__main__":
    n_moire = [30,32,34,36,48]
    theta_list = np.array([1.0845490491577974,1.0178111944464956,0.9588104855246786,0.9062751788951765,0.68204831122116])
    n_g= 5
    n_k = 10
    engine=EngineType.TBPLW
    disp=False
    valley=ValleyType.VALLEYK1
    valleyint = 1
    mk_ham_files = [glob.glob("BM_Ham/BM_Ham_MK_relaxed_theta_*.npz",recursive=True),glob.glob("BM_Ham/BM_Ham_MK_unrelaxed_theta_*.npz",recursive=True)]
    letb_ham_files = [glob.glob("BM_Ham/BM_Ham_LETB_relaxed_theta_*.npz",recursive=True),glob.glob("BM_Ham/BM_Ham_LETB_unrelaxed_theta_*.npz",recursive=True)]
    ham_files = [mk_ham_files,letb_ham_files]
    relax_type = ["relax","unrelax"]
    models = ["MK","LETB"]
    theta = np.zeros(len(n_moire))
    for m_ind,m in enumerate(models):
        for type_ind,file_list in enumerate(ham_files[m_ind]): 
            for i,ham_file in enumerate(file_list):
                theta = ham_file.split("_")[-1]
                theta = float(theta.split(".npz")[0])
                theta_ind = np.argmin(np.abs(theta-theta_list))
                n = n_moire[theta_ind]
                print(100*"=")
                print("")
                print("Theta = ",str(theta)," nmoire = ",n," model = ",m)
                print("")
                print(100*"_")
                theta_str = str(np.round(theta,decimals=2))
                bm_ham = np.load(ham_file)["bm_ham"]
                nkp = np.shape(bm_ham)[0]

                (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n)
                # set up g list
                o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
                g_vec_list = mc._set_g_vec_list_valley(n, o_g_vec_list, m_basis_vecs, valleyint)
                kmesh = mgk.set_kmesh(int(np.sqrt(nkp)), m_basis_vecs)
                T1, T2, T3, g_T1, g_T2, g_T3 = get_moire_transfer_mat(bm_ham[0,:,:],g_vec_list,m_basis_vecs,valley=valleyint,basis="TAPW")
                max_T1_k = [np.max(T1).real]
                max_T2_k = [np.max(T2).real]
                max_T3_k = [np.max(T3).real]

                T1 /= nkp
                T2 /= nkp
                T3 /= nkp

                for j in range(1,nkp):
                    T1_k, T2_k, T3_k, g_T1, g_T2, g_T3 = get_moire_transfer_mat(bm_ham[j,:,:],g_vec_list,m_basis_vecs,valley=valleyint,basis="TAPW")
                    max_T1_k.append(np.max(T1_k).real)
                    max_T2_k.append(np.max(T2_k).real)
                    max_T3_k.append(np.max(T3_k).real)
                    T1 += T1_k/nkp
                    T2 += T2_k/nkp
                    T3 += T3_k/nkp
                
                print("<T1> = ",np.mean(T1.real,axis=0), "; sigma(T1)= ",np.std(T1.real,axis=0))
                print("<T2> = ",np.mean(T2.real,axis=0), "; sigma(T2)= ",np.std(T2.real,axis=0))
                print("<T3> = ",np.mean(T3.real,axis=0), "; sigma(T3)= ",np.std(T3.real,axis=0))

                plt.scatter(kmesh[:,0],kmesh[:,1],c=max_T1_k)
                plt.title(m+", "+relax_type[type_ind]+", "+r" $\theta=$"+theta_str+r" $T_{1}(G=[0,0])$")
                plt.xlabel(r"$k_{x}$")
                plt.ylabel(r"$k_{y}$")
                plt.colorbar()
                plt.savefig("figures/"+m+"_max_T1k_u1_theta_"+theta_str+"_"+relax_type[type_ind]+".png")
                plt.clf()

                plt.scatter(kmesh[:,0],kmesh[:,1],c=max_T2_k)
                plt.title(m+", "+relax_type[type_ind]+", "+r" $\theta=$"+theta_str+r" $T_{2}, u_{2}$")
                plt.xlabel(r"$k_{x}$")
                plt.ylabel(r"$k_{y}$")
                plt.colorbar()
                plt.savefig("figures/"+m+"_max_T2k_u2_theta_"+theta_str+"_"+relax_type[type_ind]+".png")
                plt.clf()

                plt.scatter(kmesh[:,0],kmesh[:,1],c=max_T3_k)
                plt.title(m+", "+relax_type[type_ind]+", "+r" $\theta=$"+theta_str+r" $T_{3}, u_{3}$")
                plt.xlabel(r"$k_{x}$")
                plt.ylabel(r"$k_{y}$")
                plt.colorbar()
                plt.savefig("figures/"+m+"_max_T3k_u3_theta_"+theta_str+"_"+relax_type[type_ind]+".png")
                plt.clf()

                plt.scatter(g_T1[:,0],g_T1[:,1],c=T1[:,0,0].real,label="u1")
                plt.title(m+", "+relax_type[type_ind]+", "+r" $\theta=$"+theta_str+r" $T_{1}, u_{1}$")
                plt.xlabel(r"$G_{x}$")
                plt.ylabel(r"$G_{y}$")
                plt.colorbar()
                plt.savefig("figures/"+m+"_T1_u1_theta_"+theta_str+"_"+relax_type[type_ind]+".png")
                plt.clf()

                plt.scatter(g_T2[:,0],g_T2[:,1],c=T2[:,0,0].real,label="u2")
                plt.title(m+", "+relax_type[type_ind]+", "+r" $\theta=$"+theta_str+r" $T_{2}, u_{2}$")
                plt.xlabel(r"$G_{x}$")
                plt.ylabel(r"$G_{y}$")
                plt.colorbar()
                plt.savefig("figures/"+m+"_T2_u2_theta_"+theta_str+"_"+relax_type[type_ind]+".png")
                plt.clf()

                plt.scatter(g_T3[:,0],g_T3[:,1],c=T3[:,0,0].real,label="u3")
                plt.title(m+", "+relax_type[type_ind]+", "+r" $\theta=$"+theta_str+r" $T_{3}, u_{3}$")
                plt.xlabel(r"$G_{x}$")
                plt.ylabel(r"$G_{y}$")
                plt.colorbar()
                plt.savefig("figures/"+m+"_T3_u3_theta_"+theta_str+"_"+relax_type[type_ind]+".png")
                plt.clf()

                plt.scatter(np.linalg.norm(g_T1,axis=1),T1[:,0,0].real,label="u1")
                plt.scatter(np.linalg.norm(g_T1,axis=1),T1[:,0,1].real,label="u'1")

                plt.scatter(np.linalg.norm(g_T2,axis=1),T2[:,0,0].real,label="u2")
                plt.scatter(np.linalg.norm(g_T2,axis=1),T2[:,0,1].real,label="u'2")

                plt.scatter(np.linalg.norm(g_T3,axis=1),T3[:,0,0].real,label="u3")
                plt.scatter(np.linalg.norm(g_T3,axis=1),T3[:,0,1].real,label="u'3")
                plt.title(m+", "+relax_type[type_ind]) #+", "+r" $\theta=$"+theta_str+r" $T_{1}, T_{2}, T_{3}$")
                plt.xlabel(r"$|G|$")
                plt.ylabel("Coupling value (eV)")
                plt.legend()
                plt.savefig("figures/"+m+"_T_theta_"+theta_str+"_norm_"+relax_type[type_ind]+".png")
                plt.clf()



