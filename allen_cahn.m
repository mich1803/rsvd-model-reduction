%% allen_cahn.m
% POD, rSVD-POD, POD-DEIM e rSVD-DEIM per l'equazione di Allen-Cahn.
%
%
% PDE:
%   y_t - alpha*Delta y - mu*(y - y^3) = 0

clear; close all; clc;
rng(42);

%% 1. Parametri del full-order model

n = 64;
N = n^2;

alpha = 0.1;
mu = 11.0;

Tfinal = 2.0;       
dt = 0.005;
snapshot_stride = 10;

fprintf("Allen-Cahn: n=%d, N=%d, dt=%g, T=%g\n", n, N, dt, Tfinal);

%% 2. Costruzione della griglia e del Laplaciano con differenze finite

% Il dominio è Omega=(0,1)^2.
% Usiamo solo i punti interni perché imponiamo condizioni al bordo
% omogenee di Dirichlet: y=0 sul bordo.
[L, h, X, Y] = laplacian_2d(n);

% L approssima Delta.
% In 1D la seconda derivata è:
%   u_xx(x_i) ~= (u_{i-1} - 2u_i + u_{i+1}) / h^2
% In 2D:
%   Delta u = u_xx + u_yy
% La matrice 2D si ottiene con prodotti di Kronecker.

%% 3. Condizione iniziale

% y0(x,y)=0.1*sin(pi*x)*sin(pi*y)
% È liscia, nulla al bordo, e quindi compatibile con le condizioni di Dirichlet.
y = 0.1 * sin(pi*X) .* sin(pi*Y);
y = y(:);              % vettorizzo la matrice n x n in un vettore N x 1

%% 4. Schema semi-implicito per Allen-Cahn

% Partiamo dalla PDE:
%   y_t = alpha*Delta y + mu*(y-y^3)
%
% Approssimazione temporale:
%   y_t ~= (y^{n+1}-y^n)/dt
%
% Scelta semi-implicita:
%   - il termine diffusive alpha*Delta y viene preso a tempo n+1;
%   - il termine non lineare mu*(y-y^3) viene preso a tempo n.
%
% Quindi:
%   (y^{n+1}-y^n)/dt = alpha*L*y^{n+1} + mu*(y^n-(y^n)^3)
%
% Moltiplico per dt:
%   y^{n+1}-y^n = dt*alpha*L*y^{n+1} + dt*mu*(y^n-(y^n)^3)
%
% Porto il termine implicito a sinistra:
%   (I - dt*alpha*L)*y^{n+1} = y^n + dt*mu*(y^n-(y^n)^3)
%
% A ogni passo temporale risolvo quindi un sistema lineare.

I = speye(N);
A = I - dt * alpha * L;

n_steps = round(Tfinal/dt);
snapshots = [];
nonlinear_snapshots = [];
times = [];

tic;
for step = 0:n_steps

    % Salvo uno snapshot ogni snapshot_stride passi.
    % Gli snapshot NON sono indipendenti: ciascuno dipende da tutti i passi
    % precedenti perché la PDE viene integrata nel tempo.
    if mod(step, snapshot_stride) == 0
        snapshots = [snapshots, y]; %#ok<AGROW>
        nonlinear_snapshots = [nonlinear_snapshots, allen_nonlinearity(y, mu)]; %#ok<AGROW>
        times = [times, step*dt]; %#ok<AGROW>
    end

    % Costruzione del termine noto dello schema semi-implicito.
    rhs = y + dt * allen_nonlinearity(y, mu);

    % Risoluzione del sistema lineare sparse:
    %   A*y_new = rhs
    y = A \ rhs;
end
full_time = toc;

S = snapshots;              % matrice degli snapshot dello stato
F = nonlinear_snapshots;    % matrice degli snapshot non lineari f(y)

fprintf("Snapshot matrix S: %d x %d\n", size(S,1), size(S,2));
fprintf("Nonlinear snapshot matrix F: %d x %d\n", size(F,1), size(F,2));
fprintf("Tempo simulazione full-order: %.3f s\n", full_time);

%% 5. Visualizzazione di alcuni snapshot full-order

figure;
idx = round(linspace(1, size(S,2), 5));
for j = 1:length(idx)
    subplot(1, length(idx), j);
    imagesc(reshape(S(:,idx(j)), n, n));
    axis image off; colorbar;
    title(sprintf("t=%.2f", times(idx(j))));
end
sgtitle("Allen-Cahn: snapshot full-order");

%% 6. POD con SVD e rSVD

% POD:
% Data S=[y_1,...,y_Ns], calcolo SVD:
%   S = U Sigma V'
% La base POD di dimensione r è Phi = U(:,1:r).
% L'errore di proiezione è:
%   ||S - Phi*Phi'*S||_F / ||S||_F

r_values = [2 5 10 15];
p_values = [0 10];
q_values = [0 1];

err_svd = zeros(size(r_values));
time_svd = zeros(size(r_values));

err_rsvd = zeros(length(r_values), length(p_values), length(q_values));
time_rsvd = zeros(length(r_values), length(p_values), length(q_values));

for ir = 1:length(r_values)
    r = r_values(ir);

    tic;
    Phi = pod_svd(S, r);
    time_svd(ir) = toc;
    err_svd(ir) = projection_error(S, Phi);

    for ip = 1:length(p_values)
        for iq = 1:length(q_values)
            p = p_values(ip);
            q = q_values(iq);

            tic;
            Phi_r = pod_rsvd(S, r, p, q);
            time_rsvd(ir,ip,iq) = toc;
            err_rsvd(ir,ip,iq) = projection_error(S, Phi_r);
        end
    end
end

figure;
semilogy(r_values, err_svd, "-ok", "LineWidth", 2, "DisplayName", "POD-SVD");
hold on;
for ip = 1:length(p_values)
    for iq = 1:length(q_values)
        semilogy(r_values, squeeze(err_rsvd(:,ip,iq)), "-o", "LineWidth", 1.5, ...
            "DisplayName", sprintf("POD-rSVD p=%d q=%d", p_values(ip), q_values(iq)));
    end
end
grid on;
xlabel("Dimensione ridotta r");
ylabel("Errore relativo di proiezione");
title("Allen-Cahn: errore POD");
legend("Location","southwest");

figure;
plot(r_values, time_svd, "-ok", "LineWidth", 2, "DisplayName", "POD-SVD");
hold on;
for ip = 1:length(p_values)
    for iq = 1:length(q_values)
        plot(r_values, squeeze(time_rsvd(:,ip,iq)), "-o", "LineWidth", 1.5, ...
            "DisplayName", sprintf("POD-rSVD p=%d q=%d", p_values(ip), q_values(iq)));
    end
end
grid on;
xlabel("Dimensione ridotta r");
ylabel("Tempo costruzione base [s]");
title("Allen-Cahn: tempo costruzione base POD");
legend("Location","northwest");

%% 7. POD-DEIM sul termine non lineare

% DEIM approssima il termine non lineare:
%   f(y) ~= U_f * (P' U_f)^(-1) * P' f(y)
%
% U_f:
%   base POD dei nonlinear snapshots F.
%
% P':
%   operatore di selezione: prende solo alcune componenti di f(y),
%   scelte dal greedy algorithm DEIM.
%
% Qui costruiamo U_f dal 70% iniziale degli snapshot e testiamo sul 30% finale.

split = floor(0.7 * size(S,2));
F_train = F(:,1:split);
S_test  = S(:,split+1:end);

m_values = [2 5 10 15];

err_deim_svd = zeros(size(m_values));
time_deim_svd = zeros(size(m_values));

err_deim_rsvd = zeros(length(m_values), length(p_values), length(q_values));
time_deim_rsvd = zeros(length(m_values), length(p_values), length(q_values));

for im = 1:length(m_values)
    mdeim = m_values(im);

    tic;
    U_f = pod_svd(F_train, mdeim);
    time_deim_svd(im) = toc;
    indices = deim_indices(U_f);
    err_deim_svd(im) = mean_deim_error_allen(S_test, U_f, indices, mu);

    for ip = 1:length(p_values)
        for iq = 1:length(q_values)
            p = p_values(ip);
            q = q_values(iq);

            tic;
            U_fr = pod_rsvd(F_train, mdeim, p, q);
            time_deim_rsvd(im,ip,iq) = toc;
            indices_r = deim_indices(U_fr);
            err_deim_rsvd(im,ip,iq) = mean_deim_error_allen(S_test, U_fr, indices_r, mu);
        end
    end
end

figure;
semilogy(m_values, err_deim_svd, "-ok", "LineWidth", 2, "DisplayName", "POD-DEIM");
hold on;
for ip = 1:length(p_values)
    for iq = 1:length(q_values)
        semilogy(m_values, squeeze(err_deim_rsvd(:,ip,iq)), "-o", "LineWidth", 1.5, ...
            "DisplayName", sprintf("rSVD-DEIM p=%d q=%d", p_values(ip), q_values(iq)));
    end
end
grid on;
xlabel("Dimensione base DEIM m");
ylabel("Errore relativo medio su f(y)");
title("Allen-Cahn: errore DEIM sul termine non lineare");
legend("Location","southwest");

figure;
plot(m_values, time_deim_svd, "-ok", "LineWidth", 2, "DisplayName", "POD-DEIM");
hold on;
for ip = 1:length(p_values)
    for iq = 1:length(q_values)
        plot(m_values, squeeze(time_deim_rsvd(:,ip,iq)), "-o", "LineWidth", 1.5, ...
            "DisplayName", sprintf("rSVD-DEIM p=%d q=%d", p_values(ip), q_values(iq)));
    end
end
grid on;
xlabel("Dimensione base DEIM m");
ylabel("Tempo costruzione base non lineare [s]");
title("Allen-Cahn: tempo costruzione base DEIM");
legend("Location","northwest");

%% Funzioni locali

function [L,h,X,Y] = laplacian_2d(n)
    h = 1/(n+1);
    x = linspace(h, 1-h, n);
    [X,Y] = meshgrid(x,x);

    e = ones(n,1);
    T = spdiags([e -2*e e], [-1 0 1], n, n) / h^2;
    I = speye(n);

    % kron(I,T)+kron(T,I) approssima u_xx + u_yy.
    L = kron(I,T) + kron(T,I);
end

function f = allen_nonlinearity(y, mu)
    f = mu * (y - y.^3);
end

function Phi = pod_svd(S, r)
    [Phi,~,~] = svds(S, r);
end

function Phi = pod_rsvd(S, r, p, q)
    [N,Ns] = size(S);
    ell = min(r+p, min(N,Ns));

    Omega = randn(Ns, ell);
    Y = S * Omega;

    for it = 1:q
        Y = S * (S' * Y);
    end

    [Q,~] = qr(Y, 0);
    B = Q' * S;
    [Utilde,~,~] = svd(B, "econ");
    U = Q * Utilde;
    Phi = U(:,1:r);
end

function err = projection_error(S, Phi)
    Shat = Phi * (Phi' * S);
    err = norm(S - Shat, "fro") / norm(S, "fro");
end

function indices = deim_indices(U)
    % Greedy DEIM.
    % Seleziona il punto massimo del primo modo.
    [~,p0] = max(abs(U(:,1)));
    indices = p0;

    for j = 2:size(U,2)
        Uprev = U(:,1:j-1);
        pprev = indices(:);

        % Interpola il modo corrente sui punti già scelti.
        c = Uprev(pprev,:) \ U(pprev,j);

        % Residuo tra modo corrente e interpolazione.
        res = U(:,j) - Uprev*c;

        % Nuovo punto: massimo valore assoluto del residuo.
        [~,pj] = max(abs(res));
        indices = [indices; pj]; %#ok<AGROW>
    end
end

function fhat = deim_approx(f, U_f, indices)
    % Implementa:
    %   fhat = U_f * (P' U_f)^(-1) * P' f
    % In MATLAB, P'U_f equivale a U_f(indices,:)
    % e P'f equivale a f(indices).
    coeff = U_f(indices,:) \ f(indices);
    fhat = U_f * coeff;
end

function mean_err = mean_deim_error_allen(S_test, U_f, indices, mu)
    errors = zeros(1, size(S_test,2));
    for j = 1:size(S_test,2)
        y = S_test(:,j);
        f_true = allen_nonlinearity(y, mu);
        f_hat = deim_approx(f_true, U_f, indices);
        errors(j) = norm(f_true - f_hat) / max(norm(f_true), 1e-14);
    end
    mean_err = mean(errors);
end
