function [B_final,constant_term,R_max,lasso_mse,m_optimal]=lasso_regression(X,stimuli,m_min,m_max,m_step,lambda_min,lambda_max,lambda_step,ep,n,permutation)
%n equals to 3
m_interval=m_max-m_min;
% lambda_interval=lambda_max-lambda_min;
lambda_interval1 = lambda_min:(lambda_max-lambda_min)/lambda_step:lambda_max;
B_final=0;
R_max=-inf;

lasso_mse = 0;
delete(gcp('nocreate'))
for i=1:(m_step+1)
    m=m_min+(m_interval*(i-1)/m_step);
    Y=BF_sur(stimuli,ep,m,n);
    % add null distribution
    Y = Y(permutation);
%     min_mse_lasso=inf;
%     for j=1:(lambda_step+1)
%         lambda=lambda_min+(lambda_interval*(j-1)/lambda_step);
%         [B,info] = lasso(X,Y,'CV',9,'Lambda',lambda,'Intercept',true,'Standardize',false);
%         mse = info.MSE;
%         if mse<min_mse_lasso
%             min_mse_lasso = mse;
%             B_min_mse=B;
%         end
%     end
    [B,info1] = lasso(X,Y,'Lambda',lambda_interval1,'CV',5,'Options',statset('UseParallel',true));
    min_mse_lasso = info1.MSE(info1.Index1SE);
    B_min_mse = B(:,info1.Index1SE);
    R= 1 - min_mse_lasso/var(Y);
    if R>R_max
        R_max=R;
        B_final=B_min_mse;
        lasso_mse = min_mse_lasso;
        m_optimal = m;
        constant_term = info1.Intercept(info1.Index1SE);
    end
end
end