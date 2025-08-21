# code based on RES4LYF by ClownsharkBatwing (https://github.com/ClownsharkBatwing/RES4LYF)
# List of sampler names for RES4LYF
RES4LYF_NAMES = ["none",
                    "multistep/res_2m",
                    "multistep/res_3m",
                    
                    "multistep/dpmpp_2m",
                    "multistep/dpmpp_3m",

                    "multistep/abnorsett_2m",
                    "multistep/abnorsett_3m",
                    "multistep/abnorsett_4m",

                    "multistep/deis_2m",
                    "multistep/deis_3m", 
                    "multistep/deis_4m",
                    
                    "exponential/res_2s_rkmk2e", 
                    "exponential/res_2s", 
                    "exponential/res_2s_stable", 
                    "exponential/res_3s",
                    "exponential/res_3s_non-monotonic",
                    "exponential/res_3s_alt",
                    "exponential/res_3s_cox_matthews",
                    "exponential/res_3s_lie",
                    "exponential/res_3s_sunstar",

                    "exponential/res_3s_strehmel_weiner",
                    "exponential/res_4s_krogstad",
                    "exponential/res_4s_krogstad_alt",
                    "exponential/res_4s_strehmel_weiner",
                    "exponential/res_4s_strehmel_weiner_alt",
                    "exponential/res_4s_cox_matthews",
                    "exponential/res_4s_cfree4",
                    "exponential/res_4s_friedli",
                    "exponential/res_4s_minchev",
                    "exponential/res_4s_munthe-kaas",

                    "exponential/res_5s",
                    "exponential/res_5s_hochbruck-ostermann",
                    "exponential/res_6s",
                    "exponential/res_8s",
                    "exponential/res_8s_alt",

                    "exponential/res_10s",
                    "exponential/res_15s",
                    "exponential/res_16s",
                    
                    "exponential/etdrk2_2s",
                    "exponential/etdrk3_a_3s",
                    "exponential/etdrk3_b_3s",
                    "exponential/etdrk4_4s",
                    "exponential/etdrk4_4s_alt",
                    
                    "exponential/dpmpp_2s",
                    "exponential/dpmpp_sde_2s",
                    "exponential/dpmpp_3s",
                    
                    "exponential/lawson2a_2s",
                    "exponential/lawson2b_2s",

                    "exponential/lawson4_4s",
                    "exponential/lawson41-gen_4s",
                    "exponential/lawson41-gen-mod_4s",

                    "exponential/ddim",
                    
                    "hybrid/pec423_2h2s",
                    "hybrid/pec433_2h3s",
                    
                    "hybrid/abnorsett2_1h2s",
                    "hybrid/abnorsett3_2h2s",
                    "hybrid/abnorsett4_3h2s",
                    
                    "hybrid/lawson42-gen-mod_1h4s",
                    "hybrid/lawson43-gen-mod_2h4s",
                    "hybrid/lawson44-gen-mod_3h4s",
                    "hybrid/lawson45-gen-mod_4h4s",
                    
                    "linear/ralston_2s",
                    "linear/ralston_3s",
                    "linear/ralston_4s", 
                    
                    "linear/midpoint_2s",
                    "linear/heun_2s", 
                    "linear/heun_3s", 
                    
                    "linear/houwen-wray_3s",
                    "linear/kutta_3s", 
                    "linear/ssprk3_3s",
                    "linear/ssprk4_4s",
                    
                    "linear/rk38_4s",
                    "linear/rk4_4s", 
                    "linear/rk5_7s",
                    "linear/rk6_7s",

                    "linear/bogacki-shampine_4s",
                    "linear/bogacki-shampine_7s",

                    "linear/dormand-prince_6s", 
                    "linear/dormand-prince_13s", 

                    "linear/tsi_7s",

                    "linear/euler",
                    
                    "diag_implicit/irk_exp_diag_2s",
                    
                    "diag_implicit/kraaijevanger_spijker_2s",
                    "diag_implicit/qin_zhang_2s",
                    
                    "diag_implicit/pareschi_russo_2s",
                    "diag_implicit/pareschi_russo_alt_2s",
                    
                    "diag_implicit/crouzeix_2s",
                    "diag_implicit/crouzeix_3s",
                    "diag_implicit/crouzeix_3s_alt",
                    
                    "fully_implicit/gauss-legendre_2s",
                    "fully_implicit/gauss-legendre_3s", 
                    "fully_implicit/gauss-legendre_4s",
                    "fully_implicit/gauss-legendre_4s_alternating_a",
                    "fully_implicit/gauss-legendre_4s_ascending_a",
                    "fully_implicit/gauss-legendre_4s_alt",
                    "fully_implicit/gauss-legendre_5s",                    
                    "fully_implicit/gauss-legendre_5s_ascending",
                    
                    "fully_implicit/radau_ia_2s",
                    "fully_implicit/radau_ia_3s",

                    "fully_implicit/radau_iia_2s",
                    "fully_implicit/radau_iia_3s",
                    "fully_implicit/radau_iia_3s_alt",
                    "fully_implicit/radau_iia_5s",
                    "fully_implicit/radau_iia_7s",
                    "fully_implicit/radau_iia_9s",
                    "fully_implicit/radau_iia_11s",
                    
                    "fully_implicit/lobatto_iiia_2s",
                    "fully_implicit/lobatto_iiia_3s", 
                    "fully_implicit/lobatto_iiia_4s",
                    "fully_implicit/lobatto_iiib_2s",
                    "fully_implicit/lobatto_iiib_3s",
                    "fully_implicit/lobatto_iiib_4s",

                    "fully_implicit/lobatto_iiic_2s",
                    "fully_implicit/lobatto_iiic_3s",
                    "fully_implicit/lobatto_iiic_4s",

                    "fully_implicit/lobatto_iiic_star_2s",
                    "fully_implicit/lobatto_iiic_star_3s",
                    "fully_implicit/lobatto_iiid_2s",
                    "fully_implicit/lobatto_iiid_3s",                  
                    ]
