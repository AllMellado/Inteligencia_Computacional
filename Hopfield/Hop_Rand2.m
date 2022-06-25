function [N1a]=Hop_Rand2(N1,ruido,i)

if i == 1 
  N1a=vec2mat(N1,10);
  x = 1;
  y = 1; 
end
if i == 2
  N1a=vec2mat(N1,5);
  x = 2;
  y = 2; 
end
if i == 3
  N1a=vec2mat(N1,20);
  x = 3;
  y = 3; 
end

[r,c]=size(N1a); % analisa o tamanho da matriz
T=rand(r,c); % cria uma matriz rand�mica com as mesmas dimens�es de N1a 

for i=1:x:r; % cada um dos valores das matriz rand�mica s�o comparados se s�o maiores que um limiar para a troca das vari�vies
   for j=1:y:c
       if T(i,j)>=ruido % se s�o maiores, h� a troca de -1 por 1 e vice-versa
           if N1a(i,j)==-1
               N1a(i,j)=1;           
           elseif N1a(i,j)==1
               N1a(i,j)=-1; % e a matriz fica corrupta  
           end   
       end
   end
 end
end
