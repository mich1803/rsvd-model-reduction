%% image_compression.m
% compressione di immagini con SVD classica e rSVD.
%


clear; close all; clc;
rng(42);                 % seed fisso per rendere la rSVD riproducibile

%% 1. Caricamento/preparazione della matrice immagine A

if exist("monkey.jpg", "file")
    img = imread("monkey.jpg");
else
    fprintf("Immagine 'monkey.jpg' non trovata.\n");
end

% Conversione robusta in bianco e nero double in [0,1].
img = double(img);
if ndims(img) == 3
    img = 0.2989*img(:,:,1) + 0.5870*img(:,:,2) + 0.1140*img(:,:,3);
end
if max(img(:)) > 1
    img = img / 255;
end
A = img;

[m,n] = size(A);
fprintf("Dimensione matrice immagine: %d x %d\n", m, n);

figure;
imagesc(A); colormap gray; axis image off;
title("Immagine originale");

%% 2. Parametri dell'esperimento

% Valori più piccoli rispetto alla presentazione per rendere lo script veloce.
k_values = [5 25 50];
k_values = k_values(k_values < min(size(A)));

% Oversampling p: la rSVD usa ell = k+p vettori casuali.
% p serve a ridurre il rischio che il sottospazio casuale perda direzioni
% singolari importanti.
p_values = [5 15];

% Power iterations q: amplificano le direzioni associate ai valori singolari
% dominanti. q=0 è più veloce, q=1 spesso è più accurato.
q_values = [0 1];

%% 3. Esperimento: SVD classica vs rSVD

err_svd  = zeros(size(k_values));
time_svd = zeros(size(k_values));

err_rsvd  = zeros(length(k_values), length(p_values), length(q_values));
time_rsvd = zeros(length(k_values), length(p_values), length(q_values));

for ik = 1:length(k_values)
    k = k_values(ik);

    % --- SVD classica troncata ---
    % La SVD scrive A = U*S*V'.
    % Tenendo solo i primi k valori singolari otteniamo:
    % A_k = U_k*S_k*V_k'
    % che è la migliore approssimazione di rango k in norma di Frobenius.
    tic;
    [Ak_svd, ~, ~, ~] = truncated_svd(A, k);
    time_svd(ik) = toc;
    err_svd(ik) = norm(A - Ak_svd, "fro") / norm(A, "fro");

    % --- Randomized SVD ---
    for ip = 1:length(p_values)
        for iq = 1:length(q_values)
            p = p_values(ip);
            q = q_values(iq);

            tic;
            [Ak_rsvd, ~, ~, ~] = randomized_svd(A, k, p, q);
            time_rsvd(ik,ip,iq) = toc;
            err_rsvd(ik,ip,iq) = norm(A - Ak_rsvd, "fro") / norm(A, "fro");
        end
    end
end

%% 4. Plot: errore e tempi

figure;
semilogy(k_values, err_svd, "-ok", "LineWidth", 2, "DisplayName", "SVD");
hold on;
for iq = 1:length(q_values)
    for ip = 1:length(p_values)
        semilogy(k_values, squeeze(err_rsvd(:,ip,iq)), "-o", "LineWidth", 1.5, ...
            "DisplayName", sprintf("rSVD p=%d, q=%d", p_values(ip), q_values(iq)));
    end
end
grid on;
xlabel("Rango k");
ylabel("Errore relativo Frobenius");
title("Compressione immagine: errore di ricostruzione");
legend("Location","southwest");

figure;
plot(k_values, time_svd, "-ok", "LineWidth", 2, "DisplayName", "SVD");
hold on;
for iq = 1:length(q_values)
    for ip = 1:length(p_values)
        plot(k_values, squeeze(time_rsvd(:,ip,iq)), "-o", "LineWidth", 1.5, ...
            "DisplayName", sprintf("rSVD p=%d, q=%d", p_values(ip), q_values(iq)));
    end
end
grid on;
xlabel("Rango k");
ylabel("Tempo [s]");
title("Compressione immagine: tempo di calcolo");
legend("Location","northwest");

%% 5. Esempio visivo di ricostruzione

k_show = k_values(min(3, length(k_values)));
[Ak_svd, ~, ~, ~] = truncated_svd(A, k_show);
[Ak_rsvd, ~, ~, ~] = randomized_svd(A, k_show, p_values(end), 1);

figure;
subplot(1,3,1);
imagesc(A); colormap gray; axis image off;
title("Originale");

subplot(1,3,2);
imagesc(Ak_svd); colormap gray; axis image off;
title(sprintf("SVD, k=%d", k_show));

subplot(1,3,3);
imagesc(Ak_rsvd); colormap gray; axis image off;
title(sprintf("rSVD, k=%d", k_show));

%% Funzioni locali

function [Ak, U, s, V] = truncated_svd(A, k)

    k = min(k, min(size(A)) - 1);

    % Calcola i primi k valori singolari più grandi.
    % 'largest' indica che vogliamo le componenti dominanti.
    [U, S, V] = svds(A, k, "largest");

    % Estrae i valori singolari dalla matrice diagonale S.
    s = diag(S);

    % svds non garantisce sempre l'ordinamento decrescente.
    % Per coerenza con la SVD classica ordiniamo:
    % sigma_1 >= sigma_2 >= ... >= sigma_k.
    [s, idx] = sort(s, "descend");
    U = U(:, idx);
    V = V(:, idx);

    % Ricostruzione troncata:
    % A_k = U_k * Sigma_k * V_k'
    Ak = U * diag(s) * V';

    % Nel caso della compressione immagini, i valori devono rimanere
    % nell'intervallo [0,1]. Dopo la ricostruzione possono comparire
    % piccoli valori negativi o maggiori di 1 per effetto numerico.
    Ak = min(max(Ak, 0), 1);
end

function [Ak, U, s, V] = randomized_svd(A, k, p, q)
    % Randomized SVD con matrice gaussiana.
    %
    % Obiettivo: trovare rapidamente un sottospazio Q che approssimi il range
    % dominante di A, poi fare una SVD piccola su B = Q'*A.
    %
    % Input:
    %   A = matrice da comprimere
    %   k = rango target
    %   p = oversampling
    %   q = numero di power iterations

    [m,n] = size(A);
    ell = min(k+p, min(m,n));       % dimensione del sottospazio campionato

    % 1) Matrice casuale gaussiana Omega.
    %    Ha n righe perché moltiplichiamo A*Omega.
    Omega = randn(n, ell);

    % 2) Campionamento del range dominante di A.
    Y = A * Omega;

    % 3) Power iterations:
    %    Y <- A*(A'*Y)
    %    Questo amplifica le direzioni associate ai valori singolari grandi.
    for it = 1:q
        Y = A * (A' * Y);
    end

    % 4) Ortonormalizzazione QR.
    %    Le colonne di Q formano una base ortonormale del sottospazio campionato.
    [Q,~] = qr(Y, 0);

    % 5) Proiezione della matrice originale nel sottospazio ridotto.
    B = Q' * A;

    % 6) SVD piccola su B, che ha solo ell righe.
    [Utilde, Ssmall, Vfull] = svd(B, "econ");

    % 7) Recupero dei vettori singolari sinistri approssimati di A.
    Ufull = Q * Utilde;

    U = Ufull(:,1:k);
    s = diag(Ssmall);
    s = s(1:k);
    V = Vfull(:,1:k);

    Ak = U * diag(s) * V';
    Ak = min(max(Ak,0),1);
end
