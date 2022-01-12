#include "cov6d.h"
           
extern "C"{
    kernel_block get_filters(cov_matrix input);
}

//OUTPUT ALL PATCHES IN A VECTOR<MATRIXXD>, EACH ELEMENT IS A ND PATCH, EACH MATRIX IS A CONCATENATED 2D PATCH 
kernel_block get_filters(cov_matrix input){

    // cout << "------------------- Init C++ Generation of kernels -----------------" << endl;
    // cout << "COV MATRIX in c++: H: " << input.H << " W: " << input.W << " C: " << input.C << endl;
    // cout << " kernel_size: " << input.kernel_size << " stride: " << input.stride << endl;

    //Define Parameters
    int channels=input.C;
    int width=input.W;
    int height=input.H;

    int pw=input.kernel_size;
    int ph=input.kernel_size;

    int stride=input.stride;

    int r1 = input.r1;
    int r2 = input.r2;
    int r3 = input.r3;

    // cout << r1 << " " << r2 << " " << r3 << endl;

   // Rebuild matrix S
    MatrixXd S = rebuildSMatrix(input);

    // cout << "S = " << endl << S.block(0,0,25,25) << endl << endl;

//********************************************

    int pch=channels;    

//UNFOLDING MODO 1
    
    MatrixXd S1(ph,pw*ph*pch*pw*pch);// S1 = ||| |||   ||| |||  ||| |||


    for(int i=0;i<pw*pch;i++){// sweep rows of S
        S1.block(0,i*pw*ph*pch,ph,pw*ph*pch) = 
        S.block(i*ph,0,ph,pw*ph*pch);
    }

    //cout<< S1.transpose()<<endl;

//COMPUTE U1
// cout<<"computeU1"<<endl;
    BDCSVD < MatrixXd > svd(S1, Eigen::ComputeThinU);
    MatrixXd U1;
    U1=svd.matrixU();//elegir primeras columnas
    // cout<<"computeU1 end"<<endl;

//    VectorXd sigma1;
//    sigma1=svd.singularValues();
//    cout<<sigma1<<endl;

//imprimir los valores singulares

//COMPUTE U1TxS1
    MatrixXd U1TxS1;

    U1TxS1=U1.transpose()*S1; // U1TxS1=(m x n)(o x p )=(m x p) ||| |||   ||| |||   ||| |||
    //S1 ph x pw*ph*pch*pw*pch   //   4 x 324
    //U1 4 x 4

    // cout<<"U1TxS1 "<< U1TxS1.rows() << "x" << U1TxS1.cols() << endl;
    // // 4 x 324
    // cout<<"U1" << U1.rows() << "x" << U1.cols() <<endl;

//reconstruir U1TxS1 ---> "S"
/*
                                            ||| |||
    ||| |||   ||| |||   ||| |||   --->      ||| |||
                                            ||| |||
*/
// U1TxS1 ---> "tensor"

    MatrixXd U1S1tensor(ph*pw*pch,ph*pw*pch);

    for(int i=0;i<pw*pch;i++){// sweep rows of S
        U1S1tensor.block(i*ph,0,ph,pw*ph*pch) = 
        U1TxS1.block(0,i*pw*ph*pch,ph,pw*ph*pch);
    }

//
//U1S1x3 UNFOLDING MODO 3
    MatrixXd U1S1x3(pw,pw*ph*pch*ph*pch);
    
    for(int ll=0;ll<pch;ll++){    
        for(int kk=0;kk<ph;kk++){//recorre filas dentro de cuadro verde    
            for(int jj=0; jj<pw*ph*pch;jj++){
                for(int ii=0;ii<pw ; ii++){
                     U1S1x3(ii,jj+kk*pw*ph*pch+ll*pw*ph*pch*ph)=U1S1tensor(ii*ph+kk+ll*pw*ph,jj);
                }
            }
        }
    }


//    Mat cvaux;
//    eig2cv(S1,cvaux);
//    cvaux.convertTo(cvaux,0);
//    resize(cvaux)

//UNFOLDING MODO 3
    MatrixXd S3(pw,pw*ph*pch*ph*pch);
    
    for(int ll=0;ll<pch;ll++){//RECORRE CUADROS VERDES    
        for(int kk=0;kk<ph;kk++){//recorre filas dentro de cuadro AMARILLO    
            for(int jj=0; jj<pw*ph*pch;jj++){//barre todas las columnas de S
                for(int ii=0;ii<pw ; ii++){//barre el # de columnas // sacamos el vector formado por elementos rojos
                     S3(ii,jj+kk*pw*ph*pch+ll*pw*ph*pch*ph)=S(ii*ph+kk+ll*pw*ph,jj);
                }
//                cout<<"inicia"<<jj<<endl<<v<<endl;

            }
        }
    }
    //cout<< S3.transpose()<<endl;

//DESCRIPCION UF MODO 3
//PARTIMOS EN LA PRIMERA FILA DE S
//CREAMOS VECTOR MODO 3 FORMADO POR ELEMENTOS ROJOS
//RECORREMOS TODAS LAS COLUMNAS DE S
//CAMBIAMOS DE FILA EN S, HASTA LLEGAR A LA FILA "PH"
//CAMBIAMOS DE CUADRO VERDE/CHANNELS


//COMPUTE U3
// cout<<"computeU3"<<endl;
    BDCSVD < MatrixXd > svd3(S3, Eigen::ComputeThinU);
    MatrixXd U3;
    U3=svd3.matrixU();

// cout<<"computeU3 end"<<endl;

//COMPUTE U3T x3 (U1TxS1)3
//A3=(U1TxS1)3

    MatrixXd U3TA3;

    U3TA3=U3.transpose()*U1S1x3;

// U1S1x3 ESTA EN MODO 3
// U3TA3 esta en modo 3

//PASAR EL RESULTADO U3TA3 (modo 3) A TENSOR U3TA3tensor

    MatrixXd U3TA3tensor(ph*pw*pch,ph*pw*pch);
    
    for(int ll=0;ll<pch;ll++){//RECORRE CUADROS VERDES VERTICALES   
        for(int kk=0;kk<ph;kk++){//recorre filas dentro de cuadro AMARILLO    
            for(int jj=0; jj<pw*ph*pch;jj++){//barre todas las columnas de S
                for(int ii=0;ii<pw ; ii++){//barre el # de columnas // sacamos el vector formado por elementos rojos
                     U3TA3tensor(ii*ph+kk+ll*pw*ph,jj)=U3TA3(ii,jj+kk*pw*ph*pch+ll*pw*ph*pch*ph); 

//                     S3(ii,jj+kk*pw*ph*pch+ll*pw*ph*pch*ph)=S(ii*ph+kk+ll*pw*ph,jj);
                }
//                cout<<"inicia"<<jj<<endl<<v<<endl;

            }
        }
    }

//PASAR TENSOR U3TA3tensor A MODO 5 U3TA3x5
//REVISAR DIMENSIONES

    MatrixXd U3TA3x5(pch,pw*ph*pw*ph*pch);

    //VectorXd v5(pw);
    //sacar vectores formados por puntos rojos
    
    //for(int ll=0;ll<pch;ll++){    
        for(int kk=0;kk<pw*ph*pch;kk++){//recorre filas dentro de cuadro verde    
            for(int jj=0; jj<pw*ph;jj++){
                for(int ii=0;ii<pch ; ii++){
                     U3TA3x5(ii,jj+kk*pw*ph)=U3TA3tensor(ii*pw*ph+jj,kk);
                }
//                cout<<"inicia"<<jj<<endl<<v<<endl;

            }
        }

//*****************************************************************************************

//UNFOLDING MODO 5
    MatrixXd S5(pch,pw*ph*pw*ph*pch);

    //VectorXd v5(pw);
    //sacar vectores formados por puntos rojos
    
    //for(int ll=0;ll<pch;ll++){    
        for(int kk=0;kk<pw*ph*pch;kk++){//recorre filas dentro de cuadro verde    
            for(int jj=0; jj<pw*ph;jj++){
                for(int ii=0;ii<pch ; ii++){
                     S5(ii,jj+kk*pw*ph)=S(ii*pw*ph+jj,kk);
                }
//                cout<<"inicia"<<jj<<endl<<v<<endl;

            }
        }
    //}
//    cout<< S5.transpose()<<endl;

//COMPUTE U5
// cout<<"computeU5"<<endl;
    BDCSVD < MatrixXd > svd5(S5, Eigen::ComputeThinU);
    MatrixXd U5;
    U5=svd5.matrixU();
// cout<<"computeU5 end"<<endl;
//

//COMPUTE U5T x5 U3TA3x5
//A5=U3TA3x5

    MatrixXd U5TA5;

    U5TA5=U5.transpose()*U3TA3x5;

    //cout << "The matrix U5TA5 is of size "<< U5TA5.rows() << "x" << U5TA5.cols() << std::endl;

//PASAR U5TA5 A TENSOR

    MatrixXd W(pw*ph*pch,pw*ph*pch); //THIS IS TENSOR SPACE

    //VectorXd v5(pw);
    //sacar vectores formados por puntos rojos
    
    //for(int ll=0;ll<pch;ll++){    
        for(int kk=0;kk<pw*ph*pch;kk++){//recorre filas dentro de cuadro verde    
            for(int jj=0; jj<pw*ph;jj++){
                for(int ii=0;ii<pch ; ii++){
                     W(ii*pw*ph+jj,kk)=U5TA5(ii,jj+kk*pw*ph);
                }
//                cout<<"inicia"<<jj<<endl<<v<<endl;
            }
        }
    //}
//    cout<< S5.transpose()<<endl;

//COMO HICE EL UNFOLDING DE LOS CANALES - listo

//******************************************************
RowVectorXd w(pw*ph*pch);


// cout<<"antesblock"<<endl;
//extraer bloque
MatrixXd tensorspace;
//namedWindow("tensorspaces",WINDOW_NORMAL);
Mat resulth;
Mat resultv;

double normvalue;

int idx;

//Total number of kernels
int total_kns = r3*r2*r1;

// cout << "Numero de kernels " << total_kns << endl;

//Creating kernel block with the total number of tensor
kernel_tensor *kn_block_data = new kernel_tensor[total_kns];
int cnt_kn = 0;

for(int kk=0; kk<r3; kk++){
    for(int ll=0; ll<r2; ll++){
        for(int ii=0; ii<r1; ii++){
            idx=ii + ll*ph + kk*ph*pw;
            //cout<< "Mult: " << ii << " + " << ll * ph << " + " << kk * ph * pw << endl;
            //cout<< "IDX: " <<idx<<endl;
            //cout<< "PW*PH*PCH: " << pw*ph*pch << endl;

            w=W.block(idx,0,1,pw*ph*pch);//row from W
            normvalue=w.norm();
            // cout<<normvalue<<endl;

            //tensorspace = Map<MatrixXd>(W.row(ii).data(),ph,pw*pch);
            
            tensorspace=Map<MatrixXd> (w.data(), ph,pw*pch);
            tensorspace = tensorspace/normvalue;
            
            MatrixXd aux;
            vector<Mat> vectorcv(pch);

            for(int jj=0;jj<channels;jj++){
                //sacar cada bloque a matrixxd    
                aux=tensorspace.block(0,jj*pw,ph,pw);

                //convertir y guardar en vector de mat
                eigen2cv(aux,vectorcv[jj]);

                double min, max;
                cv::minMaxLoc(vectorcv[jj], &min, &max);

                vectorcv[jj]=(vectorcv[jj]-min)/(max-min)*255;
                vectorcv[jj].convertTo(vectorcv[jj],0);

                // vectorcv[jj]=((vectorcv[jj])/(max))*255;
                // vectorcv[jj].convertTo(vectorcv[jj],0);

            }

            Mat result;

            merge(vectorcv, result);
            resize(result, result, Size(320,320));

            // cout << "guardando kernels" << "--------dir" << "../cpp_results/kernel_" + to_string(cnt_kn) + ".jpg" << endl;
            // imwrite( "cpp_results/kernel_" + to_string(cnt_kn) + ".jpg", result);

            //Store one kernel
            // kernel_tensor kn = mats2kernel(vectorcv,cnt_kn);
            
            int kn_size = vectorcv[0].rows;
            int ch = vectorcv.size();

            kernel_tensor kn = matrix2kernel( tensorspace, kn_size, ch);

            //if (cnt_kn == 23 || cnt_kn == 24){
            // cout << "Tensor Space: Max: " << tensorspace.maxCoeff() << " ";
            // cout << "Min: " << tensorspace.minCoeff() << endl;

            //cout << "Tensor Space "<< cnt_kn << " = " << endl << tensorspace << endl << endl;
            
            //}
            // if (cnt_kn == 0){ print_kernel(kn, 20);}

            kn_block_data[cnt_kn] = kn;

            //Store cpp results
            // imwrite( "../cpp_results/kernel_" + to_string(cnt_kn) + ".jpg", result);

            cnt_kn += 1;
        }
    }
}
    //Create a kernel block to return the data
    kernel_block kn_block = {total_kns, kn_block_data};

    // cout << "------------------- End C++ Calculation -----------------" << endl;

    return kn_block;
}
    
