%% Comparision between ExtraPush and SubgradientPush for Least squares Regression
clc; clear; close all;
% minimize F(x) = f1(x)+f2(x)+f3(x)+f4(x)+f5(x)
% f_i=0.5*\|B_ix-b_i\|^2
% Network: 5 nodes, strongly connected
%
rng(9001);
a = norminv(rand(3,5),0,1); %sanity test to verify random number generation is the same 
%% Generating objective functions
m = 100; p = 256; k = 40;
d = randperm(p);
xs = zeros(p,1);    % true signal
xs(d(1:k)) = 2*norminv(rand(k,1),0,1);

sigma = 0.01;
B1 = 0.1*norminv(rand(m,p),0,1); b1 = B1*xs+sigma/sqrt(m)*norminv(rand(m,1),0,1);
B2 = 0.1*norminv(rand(m,p),0,1); b2 = B2*xs+sigma/sqrt(m)*norminv(rand(m,1),0,1);
B3 = 0.1*norminv(rand(m,p),0,1); b3 = B3*xs+sigma/sqrt(m)*norminv(rand(m,1),0,1);
B4 = 0.1*norminv(rand(m,p),0,1); b4 = B4*xs+sigma/sqrt(m)*norminv(rand(m,1),0,1);
B5 = 0.1*norminv(rand(m,p),0,1); b5 = B5*xs+sigma/sqrt(m)*norminv(rand(m,1),0,1);

%% Mixing matrices of the connected network
% Mixing matrix
A = [1/4,1/4,0,1/2,0; 1/4, 1/4, 0, 0, 1/3; 1/4, 0, 1/2, 0, 1/3; 0, 1/4, 0, 1/2, 0; 1/4, 1/4,1/2,0,1/3];
n = size(A,1);
A1 = (A+eye(n))/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = (B1'*B1+B2'*B2+B3'*B3+B4'*B4+B5'*B5);
b = (B1'*b1+B2'*b2+B3'*b3+B4'*b4+B5'*b5);

Opt_x = B\b;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MaxIter = 10000;
%% Initialization of EXTRA-Push %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step size parameter for EXTRA-Push
alpha0 = 0.1; % effective for the first mixing matrix
alpha1 = 0.02;

w0 = ones(n,p); % initialization of w-sequence
z0 = norminv(rand(n,p),0,1); % initialization of sequence-z via random way
x0 = z0;
w1 = A*w0;

grad01 = LS_grad(x0(1,:).', B1, b1);
grad02 = LS_grad(x0(2,:).', B2, b2);
grad03 = LS_grad(x0(3,:).', B3, b3);
grad04 = LS_grad(x0(4,:).', B4, b4);
grad05 = LS_grad(x0(5,:).', B5, b5);

myfunMD_grad0 = [grad01.';grad02.';grad03.';grad04.';grad05.'];
z00 = z0;
z01 = z0;

z10 = A*z0 - alpha0*myfunMD_grad0;
x10 = z10./w1;

z11 = A*z0 - alpha1*myfunMD_grad0;
x11 = z11./w1;

myfunMD_grad00 = myfunMD_grad0;
myfunMD_grad01 = myfunMD_grad0;

zk0 = zeros(n,p);
zk1 = zeros(n,p);

MSE_Sum0 = zeros(MaxIter,1);
Dist_Grad0 = zeros(MaxIter,1);

MSE_Sum1 = zeros(MaxIter,1);
Dist_Grad1 = zeros(MaxIter,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization of Normalized ExtraPush %%%%%%%%%%%%%%%%%%%%%
AA = A^100;
phi = AA(:,1);
nalpha0 = 0.1; % 
nalpha1 = 0.02;

nz0 = norminv(rand(n,p),0,1);
nx0 = diag(n*phi)^(-1)*nz0;

ngrad01 = LS_grad(nx0(1,:).', B1, b1);
ngrad02 = LS_grad(nx0(2,:).', B2, b2);
ngrad03 = LS_grad(nx0(3,:).', B3, b3);
ngrad04 = LS_grad(nx0(4,:).', B4, b4);
ngrad05 = LS_grad(nx0(5,:).', B5, b5);

nmyfunMD_grad0 = [ngrad01.';ngrad02.';ngrad03.';ngrad04.';ngrad05.'];

nz00 = nz0;
nz01 = nz0;

nz10 = A*nz00 - nalpha0*nmyfunMD_grad0;
nx10 = diag(n*phi)^(-1)*nz10;

nz11 = A*nz01 - nalpha1*nmyfunMD_grad0;
nx11 = diag(n*phi)^(-1)*nz11;

nmyfunMD_grad00 = nmyfunMD_grad0;
nmyfunMD_grad01 = nmyfunMD_grad0;

nzk0 = zeros(n,p);
nzk1 = zeros(n,p);

MSE_nSum0 = zeros(MaxIter,1);
Dist_nGrad0 = zeros(MaxIter,1);

MSE_nSum1 = zeros(MaxIter,1);
Dist_nGrad1 = zeros(MaxIter,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization of Subgradient-push %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gz0 = z0;
Gx0 = (A*Gz0)./w1;
Galpha = 0.1;

Ggrad01 = LS_grad(Gx0(1,:).', B1, b1);
Ggrad02 = LS_grad(Gx0(2,:).', B2, b2);
Ggrad03 = LS_grad(Gx0(3,:).', B3, b3);
Ggrad04 = LS_grad(Gx0(4,:).', B4, b4);
Ggrad05 = LS_grad(Gx0(5,:).', B5, b5);

GmyfunMD_grad0 = [Ggrad01.';Ggrad02.';Ggrad03.';Ggrad04.';Ggrad05.'];

Gz1 = A*Gz0 - Galpha*GmyfunMD_grad0;
Gx1 = Gz1./(A*w1);

Gzk = zeros(n,p);
MSE_GSum = zeros(MaxIter,1);
Dist_GGrad = zeros(MaxIter,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=1:MaxIter      
    %% Start running EXTRA-Push iteration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step size1: alpha = 0.001
    grad110 = LS_grad(x10(1,:).', B1, b1);
    grad120 = LS_grad(x10(2,:).', B2, b2);
    grad130 = LS_grad(x10(3,:).', B3, b3);
    grad140 = LS_grad(x10(4,:).', B4, b4);
    grad150 = LS_grad(x10(5,:).', B5, b5);
    
    myfunMD_grad10 = [grad110.';grad120.';grad130.';grad140.';grad150.'];
    
    Dist_Grad0(k) = norm(ones(n,1).'*myfunMD_grad10);
    
    zk0 = 2*A1*z10 - A1*z00 -alpha0*(myfunMD_grad10-myfunMD_grad00);    
    wk = A*w1;
    xk0 = zk0./wk;
    MSE_Sum0(k) = norm(xk0-repmat(Opt_x.',n,1));
    
%     w1=wk;    
    z00 = z10;
    z10 = zk0;    
    x00 = x10;
    x10 = xk0;
    
    myfunMD_grad00 = myfunMD_grad10;
    
    %% Step size2: alpha = 0.0001
    grad111 = LS_grad(x11(1,:).', B1, b1);
    grad121 = LS_grad(x11(2,:).', B2, b2);
    grad131 = LS_grad(x11(3,:).', B3, b3);
    grad141 = LS_grad(x11(4,:).', B4, b4);
    grad151 = LS_grad(x11(5,:).', B5, b5);
    
    myfunMD_grad11 = [grad111.';grad121.';grad131.';grad141.';grad151.'];
    
    Dist_Grad1(k) = norm(ones(n,1).'*myfunMD_grad11);
    
    zk1 = 2*A1*z11 - A1*z01 -alpha1*(myfunMD_grad11-myfunMD_grad01);
    
    xk1 = zk1./wk;
    MSE_Sum1(k) = norm(xk1-repmat(Opt_x.',n,1));
    
    w1=wk;    
    z01 = z11;
    z11=zk1;    
    x01 = x11;
    x11 = xk1;    
    myfunMD_grad01 = myfunMD_grad11;    
    %%% End the iteration of EXTRA-Push %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    %%Start Runing Normalized EXTRA-Push%%%%%%%%%%%%%%%%%%%%%%%%%%  
    %%% step size 1: alpha = 0.001; %%%      
    ngrad110 = LS_grad(nx10(1,:).', B1, b1);
    ngrad120 = LS_grad(nx10(2,:).', B2, b2);
    ngrad130 = LS_grad(nx10(3,:).', B3, b3);
    ngrad140 = LS_grad(nx10(4,:).', B4, b4);
    ngrad150 = LS_grad(nx10(5,:).', B5, b5);
    
    nmyfunMD_grad10 = [ngrad110.';ngrad120.';ngrad130.';ngrad140.';ngrad150.'];
    
    Dist_nGrad0(k) = norm(ones(n,1).'*nmyfunMD_grad10);
    
    nzk0 = 2*A1*nz10 - A1*nz00 -nalpha0*(nmyfunMD_grad10-nmyfunMD_grad00);
    
    nxk0 = diag(n*phi)^(-1)*nzk0;
    MSE_nSum0(k) = norm(nxk0-repmat(Opt_x.',n,1));   
   
    nz00 = nz10;
    nz10=nzk0;
    
    nx00 = nx10;
    nx10 = nxk0;   
   
    nmyfunMD_grad00 = nmyfunMD_grad10;
    
    %%% step size 2: alpha = 0.0001; %%%      
    ngrad111 = LS_grad(nx11(1,:).', B1, b1);
    ngrad121 = LS_grad(nx11(2,:).', B2, b2);
    ngrad131 = LS_grad(nx11(3,:).', B3, b3);
    ngrad141 = LS_grad(nx11(4,:).', B4, b4);
    ngrad151 = LS_grad(nx11(5,:).', B5, b5);
    
    nmyfunMD_grad11 = [ngrad111.';ngrad121.';ngrad131.';ngrad141.';ngrad151.'];
    
    Dist_nGrad1(k) = norm(ones(n,1).'*nmyfunMD_grad11);
    
    nzk1 = 2*A1*nz11 - A1*nz01 -nalpha1*(nmyfunMD_grad11-nmyfunMD_grad01); 
    
    nxk1 = diag(n*phi)^(-1)*nzk1;
    MSE_nSum1(k) = norm(nxk1-repmat(Opt_x.',n,1));   
   
    nz01 = nz11;
    nz11=nzk1;
    
    nx01 = nx11;
    nx11 = nxk1;   
   
    nmyfunMD_grad01 = nmyfunMD_grad11;
    %%% End the iteration of Normalized EXTRA-Push %%%%%%%%%%%%%%%%%%  
    
    %% Start runing iteration of subgradient-push
    Galpha = 0.8/sqrt(k); % step size for subgradient-push
    
    Ggrad11 = LS_grad(Gx1(1,:).', B1, b1);
    Ggrad12 = LS_grad(Gx1(2,:).', B2, b2);
    Ggrad13 = LS_grad(Gx1(3,:).', B3, b3);
    Ggrad14 = LS_grad(Gx1(4,:).', B4, b4);
    Ggrad15 = LS_grad(Gx1(5,:).', B5, b5);
    
    GmyfunMD_grad1 = [Ggrad11.';Ggrad12.';Ggrad13.';Ggrad14.';Ggrad15.'];
    
    Dist_GGrad(k) = norm(ones(n,1).'*GmyfunMD_grad1);
       
    Gzk = A*Gz1 -Galpha*GmyfunMD_grad1;
    
    Gxk = Gzk./(A*wk);
    MSE_GSum(k) = norm(Gxk-repmat(Opt_x.',n,1)); 
     
    Gz1=Gzk;    
    Gx1 = Gxk;         
    GmyfunMD_grad0 = GmyfunMD_grad1;
    %%% End subgradient-push %%%%%%%%%%%%%%%%%%
end

figure,
semilogy([1:200:MaxIter]',MSE_Sum0(1:200:MaxIter),'r--','Linewidth',3);
hold on;
semilogy([1:200:MaxIter]',MSE_nSum0(1:200:MaxIter),'c-.','Linewidth',3);
hold on;
semilogy([1:200:MaxIter]',MSE_Sum1(1:200:MaxIter),'b--','Linewidth',3);
hold on;
semilogy([1:200:MaxIter]',MSE_nSum1(1:200:MaxIter),'k-.','Linewidth',3);
hold on;
semilogy([1:200:MaxIter]',MSE_GSum(1:200:MaxIter),'m--','Linewidth',3);
xlabel('Iteration t'); ylabel('Iterative Error'); %||x^t -x^*||_2/||x^0 -x^*||_2
legend('ExtraPush (\alpha=0.1)', 'Normlized ExtraPush (\alpha=0.1)', 'ExtraPush (\alpha=0.02)','Normlized ExtraPush (\alpha=0.02)','Subgradient-Push (hand optimized)');
axis([0, MaxIter, 10^(-9),10^(2)]);
