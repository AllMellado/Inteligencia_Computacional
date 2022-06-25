function [Out_rede]=Net_Hopfield_sign(Vatu,W)
epoca=0; % inicia contador de época 
aux=1; % define variável auxiliar como 1

while aux==1 % enquanto for 1, o loop while continuará a ser executado
    
    Vant = Vatu; % atualiza o valor das entradas
    %u = W*Vant; % Multiplicação pelos pesos sinápticos
    Vatu = sign(W*Vant); % camada de saída FUNÇÃO SINAL
    
    if isequal(Vant,Vatu)==1 % se a saída for igual duas vezes consecutivas, para-se o loop while
        aux=0;
    end
    
    epoca=epoca+1; % contador de época
    
    if epoca==1000 % limitador de épocas
        aux=0;
    end
end 

Out_rede=Vatu; % saída é atualizado na variável Out_rede
end
