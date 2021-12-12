function [out]=func(N,tr)
 fun1=@(x,y) (y.^N(1,3)).*(x.^N(1,2)).*((1-x-y).^N(1,1));
 fun2=@(x,y) (y.^N(2,3)).*(x.^N(2,1)).*((1-x-y).^N(2,2));
 fun3=@(x,y) (y.^N(3,1)).*(x.^N(3,2)).*((1-x-y).^N(3,3));
 ymax = @(x) 1 - x;
if tr==[1,1]
    fun=@(x,y) (y.^N(1,3)).*(x.^N(1,2)).*((1-x-y).^(N(1,1)+1));
    out=integral2(fun,0,1,0,ymax).*integral2(fun2,0,1,0,ymax).*integral2(fun3,0,1,0,ymax);
end
if tr==[1,2]
    fun=@(x,y) (y.^N(1,3)).*(x.^(N(1,2)+1)).*((1-x-y).^N(1,1));
    out=integral2(fun,0,1,0,ymax).*integral2(fun2,0,1,0,ymax).*integral2(fun3,0,1,0,ymax);
end
if tr==[1,3]
    fun=@(x,y) (y.^(N(1,3)+1)).*(x.^N(1,2)).*((1-x-y).^(N(1,1)+1));
    out=integral2(fun,0,1,0,ymax).*integral2(fun2,0,1,0,ymax).*integral2(fun3,0,1,0,ymax);
end
if tr==[2,1]
    fun=@(x,y) (y.^N(2,3)).*(x.^(N(2,1)+1)).*((1-x-y).^N(2,2));
    out=integral2(fun,0,1,0,ymax).*integral2(fun1,0,1,0,ymax).*integral2(fun3,0,1,0,ymax);
end
if tr==[2,2]
    fun=@(x,y) (y.^N(2,3)).*(x.^N(2,1)).*((1-x-y).^(N(2,2)+1));
    out=integral2(fun,0,1,0,ymax).*integral2(fun1,0,1,0,ymax).*integral2(fun3,0,1,0,ymax);
end
if tr==[2,3]
    fun=@(x,y) (y.^(N(2,3)+1)).*(x.^N(2,1)).*((1-x-y).^N(2,2));
    out=integral2(fun,0,1,0,ymax).*integral2(fun1,0,1,0,ymax).*integral2(fun3,0,1,0,ymax);
end
if tr==[3,1]
    fun=@(x,y) (y.^(N(3,1)+1)).*(x.^N(3,2)).*((1-x-y).^N(3,3));
    out=integral2(fun,0,1,0,ymax).*integral2(fun1,0,1,0,ymax).*integral2(fun2,0,1,0,ymax);
end
if tr==[3,2]
    fun=@(x,y) (y.^N(3,1)).*(x.^(N(3,2)+1)).*((1-x-y).^N(3,3));
    out=integral2(fun,0,1,0,ymax).*integral2(fun1,0,1,0,ymax).*integral2(fun2,0,1,0,ymax);
end
if tr==[3,3]
    fun=@(x,y) (y.^N(3,1)).*(x.^N(3,2)).*((1-x-y).^(N(3,3)+1));
    out=integral2(fun,0,1,0,ymax).*integral2(fun1,0,1,0,ymax).*integral2(fun2,0,1,0,ymax);
end
end