%% steady_elliptical.m
% POD, rSVD-POD, POD-DEIM e rSVD-DEIM per una PDE ellittica non lineare.
%
% PDE stazionaria:
%   -Delta u(x,y) + s(u;mu) = 100 sin(2*pi*x) sin(2*pi*y)
%
% con:
%   s(u;mu) = (mu1/mu2) * (exp(mu2*u)-1)
%
% A differenza di Allen-Cahn, qui NON c'è evoluzione temporale.
% Ogni snapshot è la soluzione full-order della PDE per una diversa coppia
% di parametri (mu1,mu2).

clear; close all; clc;
rng(42);

%% 1. Parametri del problema

% Uso una griglia più piccola della presentazione per rendere MATLAB veloce.
n = 50;
N = n^2;

newton_tol = 1e-10;
newton_max_iter = 25;

fprintf("Ellittica non lineare: n=%d, N=%d\n", n, N);

%% 2. Griglia, Laplaciano e termine noto

[L, h, X, Y] = laplacian_2d(n);

% rhs(x,y) = 100 sin(2*pi*x) sin(2*pi*y)
rhs_grid = 100 * sin(2*pi*X) .* sin(2*pi*Y);
rhs = rhs_grid(:);

%% 3. Costruzione degli snapshot parametrici

% Nella presentazione si usano 16x16 parametri.
% Qui usiamo 6x6 per avere un codice più rapido ma concettualmente identico.
mu1_values = linspace(0.01, 10.0, 6);
mu2_values = linspace(0.01, 10.0, 6);

S = [];          % snapshot delle soluzioni u
F = [];          % snapshot dei termini non lineari s(u;mu)
params = [];     % coppie (mu1,mu2)
solve_times = [];

% Newton ha bisogno di una stima iniziale.
% Per il primo parametro uso u0=0.
% Per i successivi uso la soluzione precedente come initial guess:
% questa è una continuazione parametrica semplice e accelera la convergenza.
previous_u = zeros(N,1);

fprintf("Genero snapshot full-order...\n");
for i = 1:length(mu1_values)
    for j = 1:length(mu2_values)
        mu1 = mu1_values(i);
        mu2 = mu2_values(j);

        tic;
        [u, converged, iters] = solve_elliptic_newton( ...
            L, rhs, mu1, mu2, previous_u, newton_tol, newton_max_iter);
        elapsed = toc;

        if ~converged
            fprintf("Warning: Newton non convergente per mu=(%.2f, %.2f), iter=%d\n", ...
                mu1, mu2, iters);
        end

        previous_u = u;

        S = [S, u]; %#ok<AGROW>
        F = [F, nonlinear_s(u, mu1, mu2)]; %#ok<AGROW>
        params = [params; mu1, mu2]; %#ok<AGROW>
        solve_times = [solve_times, elapsed]; %#ok<AGROW>
    end
end

fprintf("Snapshot matrix S: %d x %d\n", size(S,1), size(S,2));
fprintf("Nonlinear snapshot matrix F: %d x %d\n", size(F,1), size(F,2));
fprintf("Tempo medio soluzione full-order: %.3f s\n", mean(solve_times));

%% 4. Visualizzazione di alcune soluzioni

figure;
idx = round(linspace(1, size(S,2), 5));
for k = 1:length(idx)
    subplot(1,length(idx),k);
    imagesc(reshape(S(:,idx(k)), n, n));
    axis image off; colorbar;
    title(sprintf("\\mu=(%.1f, %.1f)", params(idx(k),1), params(idx(k),2)));
end
sgtitle("PDE ellittica: snapshot full-order");

%% 5. POD con SVD e rSVD

% POD:
%   S = U Sigma V'
%   Phi_r = U(:,1:r)
%   errore = ||S - Phi_r Phi_r' S||_F / ||S||_F
%
% SVD è il riferimento deterministico.
% rSVD cerca lo stesso sottospazio dominante usando proiezioni casuali.

r_values = [2 4 8 12];
p_values = [0 8];
q_values = [0 1];

err_svd = zeros(size(r_values));
time_svd = zeros(size(r_values));

err_rsvd = zeros(length(r_values), length(p_values), length(q_values));
time_rsvd = zeros(length(r_values), length(p_values), length(q_values));

for ir = 1:length(r_values)
    r = min(r_values(ir), min(size(S))-1);

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
title("PDE ellittica: errore POD");
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
title("PDE ellittica: tempo costruzione base POD");
legend("Location","northwest");

%% 6. POD-DEIM sul termine non lineare parametrico

% Il termine non lineare è:
%   s(u;mu) = (mu1/mu2)(exp(mu2*u)-1)
%
% Costruiamo una base U_f dai nonlinear snapshots F.
% Poi DEIM seleziona pochi indici e approssima:
%   s(u;mu) ~= U_f * (P'U_f)^(-1) * P's(u;mu)

split = floor(0.7 * size(S,2));
F_train = F(:,1:split);
S_test = S(:,split+1:end);
params_test = params(split+1:end,:);

m_values = [2 4 8 12];

err_deim_svd = zeros(size(m_values));
time_deim_svd = zeros(size(m_values));

err_deim_rsvd = zeros(length(m_values), length(p_values), length(q_values));
time_deim_rsvd = zeros(length(m_values), length(p_values), length(q_values));

for im = 1:length(m_values)
    mdeim = min(m_values(im), min(size(F_train))-1);

    tic;
    U_f = pod_svd(F_train, mdeim);
    time_deim_svd(im) = toc;
    indices = deim_indices(U_f);
    err_deim_svd(im) = mean_deim_error_elliptic(S_test, params_test, U_f, indices);

    for ip = 1:length(p_values)
        for iq = 1:length(q_values)
            p = p_values(ip);
            q = q_values(iq);

            tic;
            U_fr = pod_rsvd(F_train, mdeim, p, q);
            time_deim_rsvd(im,ip,iq) = toc;
            indices_r = deim_indices(U_fr);
            err_deim_rsvd(im,ip,iq) = mean_deim_error_elliptic(S_test, params_test, U_fr, indices_r);
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
ylabel("Errore relativo medio su s(u;mu)");
title("PDE ellittica: errore DEIM");
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
title("PDE ellittica: tempo costruzione base DEIM");
legend("Location","northwest");

%% Funzioni locali

function [L,h,X,Y] = laplacian_2d(n)
    h = 1/(n+1);
    x = linspace(h, 1-h, n);
    [X,Y] = meshgrid(x,x);

    e = ones(n,1);
    T = spdiags([e -2*e e], [-1 0 1], n, n) / h^2;
    I = speye(n);

    % L approssima Delta.
    L = kron(I,T) + kron(T,I);
end

function s = nonlinear_s(u, mu1, mu2)
    % s(u;mu) = (mu1/mu2)(exp(mu2*u)-1)
    % Clip numerico per evitare overflow dell'esponenziale.
    z = mu2*u;
    z = max(min(z, 50), -50);
    s = (mu1/mu2) * (exp(z) - 1);
end

function sp = nonlinear_s_prime(u, mu1, mu2)
    % Derivata rispetto a u:
    % d/du [(mu1/mu2)(exp(mu2*u)-1)] = mu1*exp(mu2*u)
    z = mu2*u;
    z = max(min(z, 50), -50);
    sp = mu1 * exp(z);
end

function [u, converged, iters] = solve_elliptic_newton(L, rhs, mu1, mu2, initial_guess, tol, max_iter)
    % Risolve:
    %   -Delta u + s(u;mu) = rhs
    %
    % Poiché L approssima Delta, il residuo discreto è:
    %   R(u) = -L*u + s(u;mu) - rhs
    %
    % Newton:
    %   J(u_k)*du = -R(u_k)
    %   u_{k+1} = u_k + du
    %
    % Jacobiano:
    %   J = -L + diag(s'(u;mu))

    u = initial_guess;
    converged = false;

    for iters = 1:max_iter
        R = -L*u + nonlinear_s(u, mu1, mu2) - rhs;
        rel_res = norm(R) / max(norm(rhs), 1e-14);

        if rel_res < tol
            converged = true;
            return;
        end

        J = -L + spdiags(nonlinear_s_prime(u, mu1, mu2), 0, length(u), length(u));
        du = J \ (-R);
        u = u + du;
    end
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
    [~,p0] = max(abs(U(:,1)));
    indices = p0;

    for j = 2:size(U,2)
        Uprev = U(:,1:j-1);
        pprev = indices(:);

        c = Uprev(pprev,:) \ U(pprev,j);
        res = U(:,j) - Uprev*c;

        [~,pj] = max(abs(res));
        indices = [indices; pj]; %#ok<AGROW>
    end
end

function fhat = deim_approx(f, U_f, indices)
    coeff = U_f(indices,:) \ f(indices);
    fhat = U_f * coeff;
end

function mean_err = mean_deim_error_elliptic(S_test, params_test, U_f, indices)
    errors = zeros(1, size(S_test,2));

    for j = 1:size(S_test,2)
        u = S_test(:,j);
        mu1 = params_test(j,1);
        mu2 = params_test(j,2);

        f_true = nonlinear_s(u, mu1, mu2);
        f_hat = deim_approx(f_true, U_f, indices);

        errors(j) = norm(f_true - f_hat) / max(norm(f_true), 1e-14);
    end

    mean_err = mean(errors);
end
