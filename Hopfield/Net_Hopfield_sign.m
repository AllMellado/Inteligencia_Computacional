function [Out_rede]=Net_Hopfield_sign(Vatu,W)
epoca=0; % inicia contador de �poca 
aux=1; % define vari�vel auxiliar como 1

while aux==1 % enquanto for 1, o loop while continuar� a ser executado
    
    Vant = Vatu; % atualiza o valor das entradas
    %u = W*Vant; % Multiplica��o pelos pesos sin�pticos
    Vatu = sign(W*Vant); % camada de sa�da FUN��O SINAL
    
    if isequal(Vant,Vatu)==1 % se a sa�da for igual duas vezes consecutivas, para-se o loop while
        aux=0;
    end
    
    epoca=epoca+1; % contador de �poca
    
    if epoca==1000 % limitador de �pocas
        aux=0;
    end
end 

Out_rede=Vatu; % sa�da � atualizado na vari�vel Out_rede
end
