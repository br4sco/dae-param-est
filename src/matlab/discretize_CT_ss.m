function [Ad,Bd] = discretize_CT_ss(A,B,Ts, eps)

    n = size(A,1);
    Mexp = [-A B*(B'); zeros(size(A)) A'];
    MTs = exp(Mexp*Ts);
    Ad = MTs(n+1:end, n+1:end)';
    Dd = Ad*MTs(1:n, n+1:end);
    Dd = (Dd + Dd')./2;
    try
        Bd = chol(Dd);
    catch
        warning("Dd was singular, had to be perturbed for Cholesky factorization");
        Bd = chol(Dd+eps*eye(size(Dd)));
    end
end

