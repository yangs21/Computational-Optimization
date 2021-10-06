function [w,b,out] = ALM_SVM(X,y,lam,opts)
%=============================================
% augmented Lagrangian method for solving SVM
% min_{w,b,t} sum(t) + lam/2*norm(w)^2
% s.t. y(i)*(w'*X(:,i)+b) >= 1-t(i)
%      t(i) >= 0, i = 1,...,N
%
%===============================================
%
% ==============================================
% input:
%       X: training data, each column is a sample data
%       y: label vector
%       lam: model parameter
%       opts.tol: stopping tolerance
%       opts.maxit: maximum number of outer iteration
%       opts.subtol: stopping tolerance for inner-loop
%       opts.maxsubit: maxinum number of iteration for inner-loop
%       opts.w0: initial w
%       opts.b0: initial b0
%       opts.t0: initial t0
%       opts.beta: penalty parameter
%
% output:
%       w: learned w
%       b: learned b
%       out.hist_pres: historical primal residual
%       out.hist_dres: historical dual residual
%       out.hist_subit: historical iteration number of inner-loop

% ======================================================

%% get size of problem: p is dimension; N is number of data pts
[p,N] = size(X);

%% set parameters
if isfield(opts,'tol')        tol = opts.tol;           else tol = 1e-3;       end
if isfield(opts,'maxit')      maxit = opts.maxit;       else maxit = 30;       end
if isfield(opts,'subtol')     subtol = opts.subtol;     else subtol = 1e-3;    end
if isfield(opts,'maxsubit')   maxsubit = opts.maxsubit; else maxsubit = 30;    end
if isfield(opts,'w0')         w0 = opts.w0;             else w0 = randn(p,1);  end
if isfield(opts,'b0')         b0 = opts.b0;             else b0 = 0;           end
if isfield(opts,'t0')         t0 = opts.t0;             else t0 = zeros(N,1);  end
if isfield(opts,'s0')         s0 = opts.s0;             else s0 = zeros(N,1);  end
if isfield(opts,'beta')       beta = opts.beta;         else beta = 0.5;       end

alpha_w = 0.6;
alpha_b = 0.6;
alpha_t = 0.6;
alpha_s = 0.6;
c = 0.35;
dec_ratio = 0.35;

w = w0; b = b0; t = t0; s = max(0,s0);

% initialize dual variable
u = zeros(N,1);
%% compute the primal residual and save to pres
r = ones(N,1) - t - b*y - y.*X'*w;
pres = norm(r);

% save historical primal residual
hist_pres = pres;

%% compute dual residual
temp = 0;
for i = 1:N
   temp = temp + u(i)*y(i)*X(:,i); 
end
dres = norm(w - (1/lam)*temp);

hist_dres = dres;

hist_subit = 0;

iter = 0; subit = 0;
%% start of outer loop
while max(pres,dres) > tol && iter < maxit
    iter = iter + 1;
    subiter = 0;
    
    % call the subroutine to update primal variable (w,b,t)
    w0 = w;
    b0 = b;
    t0 = t;
    s0 = s;
    
    % fill in the subsolver by yourself
    % if slack variables are introduced, you will have more variables
    [w,b,t,s] = subsolver(w0,b0,t0,s0,subtol,maxsubit);
    subit = subiter;
    hist_subit = [hist_subit; subit]; 
    % update multiplier u 
    for i = 1:N
        u(i) = max(0, u(i) + beta*(1 - t(i) - y(i)*(w'*X(:,i)+b)+s(i)));
    end
    
    % update primal residual
    r =  t - b*y - (y.*X')*w;
    pres = norm(1 - norm(r));
    
    % save pres to hist_pres
    hist_pres = [hist_pres; pres];
    
    % compute gradient of ordinary Lagrangian function about (w,b,t,s)
    temp1 = zeros(p,1);
    temp2 = 0;
    grad_t = zeros(N,1);
    grad_s = zeros(N,1);
    for i = 1:N
        temp1 = temp1 + u(i)*y(i)*X(:,i) + beta*(1-t(i)-y(i)*(w'*X(:,i)+b)+s(i))*y(i)*X(:,i);
        temp2 = temp2 + u(i)*y(i) + beta*y(i)*(1-t0(i)-y(i)*(w'*X(:,i)+b)+s(i));
        grad_t(i) = 1 - u(i) - beta*(1 - t(i) - y(i)*(w'*X(:,i)+b)+s(i));
        grad_s(i) = u(i) + beta*(1 - t(i) - y(i)*(w'*X(:,i)+b));
    end
    grad_w = lam*w - temp1;
    grad_b = -temp2;
    
    % compute the dual residual and save to hist_dres
    temp_w = 0;
    temp_b = 0;
    temp_t = 0;
    temp_s= 0;
    for i = 1:N
       temp_w = temp_w + beta*(1-t(i)-y(i)*(w'*X(:,i)+b)+s(i)) * y(i)*X(:,i);
       temp_b = temp_b + 1-t(i)-y(i)*(w'*X(:,i)+b)+s(i);
       temp_t = temp_t + beta*(1-t(i)-y(i)*(w'*X(:,i)+b)+s(i));
       temp_s = temp_s + beta*(1-t(i)-y(i)*(w'*X(:,i)+b)+s(i));
    end
    %dres = norm(temp_w,1) + norm(temp_b,1) + norm(temp_t,1);
    dres = norm(temp_w,1) + norm(temp_b,1) + norm(temp_t,1) + norm(temp_s,1);
    
    % save pres to hist_pres
    hist_dres = [hist_dres; dres];
    
    fprintf('out iter = %d, pres = %5.4e, dres = %5.4e, subit = %d\n',iter,pres,dres,subit);
end

out.hist_pres = hist_pres;
out.hist_dres = hist_dres;
out.hist_subit = hist_subit;

%% =====================================================
% subsolver for primal subproblem
    function [w,b,t,s] = subsolver(w0,b0,t0,s0,subtol,maxsubit)
    % projected penalty to solve min_w L(w,b,y,s,u)
    
    % initialization
    grad_err = 0;
    
    % compute gradient of ordinary Lagrangian function about (w,b,t,s)
    temp1 = zeros(p,1);
    temp2 = 0;
    grad_t = zeros(N,1);
    grad_s = zeros(N,1);
    for j = 1:N
        temp1 = temp1 + u(j)*y(j)*X(:,j) + beta*(1-t0(j)-y(j)*(w0'*X(:,j)+b0)+s0(j))*y(j)*X(:,j);
        temp2 = temp2 + u(j)*y(j) + beta*y(j)*(1-t0(j)-y(j)*(w0'*X(:,j)+b0)+s0(j));
        grad_t(j) = 1 - u(j) - beta*(1 - t0(j) - y(j)*(w0'*X(:,j)+b0)+s0(j));
        grad_s(j) = u(j) + beta*(1 - t0(j) - y(j)*(w0'*X(:,j)+b0)+s0(j));
    end
    grad_w = lam*w0 - temp1;
    %fprintf("grad norm = %5.4e\n",norm(grad_w,1));
    grad_b = -temp2;
    
    % compute the residual for the constraint
    temp = 0;
    for j = 1:N
        temp = temp + 1 - t0(j) - y(j)*((w0'*X(:,j)+b0)+s0);
    end
    r = temp;
    res = norm(r);
    hist_res = res;
    
    % compute objective
    temp3 = 0;
    for j = 1:N
        temp3 = temp3 + t0(j) + u(j)*(1-t0(j)-y(j)*(w0'*X(:,j)+b0)+s0(j))+0.5*beta*(1-t0(j)-y(j)*(w0'*X(:,j)+b0)+s0(j))^2;
    end
    obj = norm(0.5*lam*norm(w0,2) + temp3);
    hist_obj = obj; 

    while res > (subtol | grad_err > subtol) && (subiter < maxsubit)
        % compute stepsize of w
        p_d = -grad_w;
        w_tmp = w0 + alpha_w * p_d;                
        LF = LF_Value(N,t0,u,y,w0,X,b0,s0,beta,lam);
        LF_obj = LF_Value(N,t0,u,y,w_tmp,X,b0,s0,beta,lam);
        while LF_obj > LF + c * alpha_w * grad_w'*p_d
            alpha_w = dec_ratio * alpha_w;
            w_tmp = w0 + alpha_w * p_d;  
            LF_obj = LF_Value(N,t0,u,y,w_tmp,X,b0,s0,beta,lam);
        end
        % update w with alpha found
        w = w0 + alpha_w * p_d;   
        % compute the new gradient
        temp1 = zeros(p,1);
        for j = 1:N
            temp1 = temp1 + u(j)*y(j)*X(:,j) + beta*(1-t0(j)-y(j)*(w'*X(:,j)+b0)+s0(j))*y(j)*X(:,j);
        end
        grad_w = lam*w - temp1;
        % compute violation of optimality condition
        grad_err = norm(grad_w,1);
        %fprintf("new grad of grad_w = %5.4e\n",grad_err);

        % compute stepsize of b
        p_d = -grad_b;
        b_tmp = b0 + alpha_b * p_d;                
        LF = LF_Value(N,t0,u,y,w0,X,b0,s0,beta,lam);
        LF_obj = LF_Value(N,t0,u,y,w0,X,b_tmp,s0,beta,lam);
        while LF_obj > LF + c * alpha_b * grad_b'*p_d
            alpha_b = dec_ratio * alpha_b;
            b_tmp = b0 + alpha_b * p_d;  
            LF_obj = LF_Value(N,t0,u,y,w0,X,b_tmp,s0,beta,lam);
        end
        % update b
        b = b0 - alpha_b * grad_b;
        % compute the new gradient
        temp2 = 0;
        for j=1:N
            temp2 = temp2 + u(j)*y(j) + beta*y(j)*(1-t0(j)-y(j)*(w0'*X(:,j)+b)+s0(j));
        end
        grad_b = -temp2;
        % compute new violation of optimality condition
        grad_err = norm(grad_b,1);
        %fprintf("new grad of grad_b = %5.4e\n",grad_err);

        % compute stepsize of t
        p_d = -grad_t;
        t_tmp = t0 + alpha_t * p_d;                
        LF = LF_Value(N,t0,u,y,w0,X,b0,s0,beta,lam);
        LF_obj = LF_Value(N,t_tmp,u,y,w0,X,b0,s0,beta,lam);
        while LF_obj > LF + c * alpha_t * grad_t'*p_d
            alpha_t = dec_ratio * alpha_t;
            t_tmp = t0 + alpha_t * p_d;  
            LF_obj = LF_Value(N,t_tmp,u,y,w0,X,b0,s0,beta,lam);
        end
        %update t with the alpha
        t = t0 - alpha_t * grad_t;
        % compute the new gradient
        for j = 1:N
            grad_t(j) = 1 - u(j) - beta*(1 - t(j) - y(j)*(w0'*X(:,j)+b0) + s0(j));
        end
        % compute new violation of optimality condition
        grad_err = norm(grad_t,1);
        %fprintf("new grad of grad_t = %5.4e\n",grad_err);
        
        % compute violation of optimality condition for s
        grad_err = 0;
        for j = 1:N
            if s0(j) == 0
                grad_err = grad_err + max(0, -grad_s(j));
            else
                grad_err = grad_err + abs(grad_s(j));
            end
        end
        s = zeros(N,1);
        % compute stepsize of s
        p_d = -grad_s;
        s_tmp = s0 + alpha_s * p_d;                
        LF = LF_Value(N,t0,u,y,w0,X,b0,s0,beta,lam);
        LF_obj = LF_Value(N,t0,u,y,w0,X,b0,s_tmp,beta,lam);
        while LF_obj > LF + c * alpha_s * grad_s'*p_d
            alpha_s = dec_ratio * alpha_s;
            s_tmp = s0 + alpha_s * p_d;  
            LF_obj = LF_Value(N,t0,u,y,w0,X,b0,s_tmp,beta,lam);
        end
        % update s with alpha_s
        for j = 1:N
            s(j) = max(0, s0(j) - alpha_s * grad_s(j));
        end
        % compute the new gradient
        grad_s = zeros(N,1);
        for j = 1:N
            grad_s(j) = u(j) + beta*(1 - t0(j) - y(j)*(w0'*X(:,j)+b0) + s(j));
        end
        % compute new violation of optimality condition
        grad_err = 0;
        for j = 1:N
            if s(j) == 0
                grad_err = grad_err + max(0, -grad_s(j));
            else
                grad_err = grad_err + abs(grad_s(j));
            end
        end
        %fprintf("new grad of grad_s = %5.4e\n",grad_err);
  
        % compute the residual
        temp = 0;
        for j = 1:N
            temp = temp + (1-t(j)-y(j)*(w'*X(:,j)+b)+s(j));
        end
        r = temp;
        res = norm(r);

        % compute the residual
        %res = norm()
        
        % compute the objective
        temp = 0;
        for j = 1:N
            temp = temp + t(j) + u(j)*(1-t(j)-y(j)*(w'*X(:,j)+b)+s(j))+0.5*beta*(1-t(j)-y(j)*(w'*X(:,j)+b)+s(j))^2;
        end
        obj = norm(0.5*lam*norm(w,2) + temp);
        hist_obj = obj; 

        % save res and obj
        hist_res = [hist_res; res];
        hist_obj = [hist_obj; obj];
    
        % update multiplier
        for j = 1:N
            u(j) = max(0, u(j) + beta*(1 - t(j) - y(j)*(w'*X(:,j)+b)+s(j)));
        end
        
    subiter = subiter + 1;
    end
    
    %Function to compute Lagrange Function value
    function [obj] = LF_Value(N,t0,u,y,w0,X,b0,s0,beta,lam)
        % compute objective
        temp = 0;
        for j = 1:N
            temp = temp + t0(j) + u(j)*(1-t0(j)-y(j)*(w0'*X(:,j)+b0)+s0(j))+0.5*beta*(1-t0(j)-y(j)*(w0'*X(:,j)+b0)+s0(j))^2;
        end
        obj = 0.5*lam*norm(w0,2) + temp; 
    end
    
    end
%=====================================================
end



