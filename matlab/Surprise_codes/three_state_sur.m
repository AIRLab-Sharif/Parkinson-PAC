function [p]=three_state_sur(string,w)
%the string must be a column vector so that each of its elements is 1 or 2 or 3
%the output is a probability vector
%w is forgetting factor
N=zeros(3);
p=zeros(1,size(string,1));
for i=2:size(string,1)
    N=exp(-1/w)*N;
    N(string(i-1),string(i))=N(string(i-1),string(i))+1;
    fun1=@(x,y) (y.^N(1,3)).*(x.^N(1,2)).*((1-x-y).^N(1,1));
    fun2=@(x,y) (y.^N(2,3)).*(x.^N(2,1)).*((1-x-y).^N(2,2));
    fun3=@(x,y) (y.^N(3,1)).*(x.^N(3,2)).*((1-x-y).^N(3,3));
    ymax = @(x) 1 - x;
    denominator=integral2(fun1,0,1,0,ymax).*integral2(fun2,0,1,0,ymax).*integral2(fun3,0,1,0,ymax);
    tr=[string(i-1),string(i)];
    p(i)=func(N,tr)./denominator;
end