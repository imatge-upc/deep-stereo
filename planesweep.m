%% Parameters
%inname  = 'ballet';
inname  = 'Dancer';
%inname = 'balloons';
views = [3];
frame = 0;
%calibinname = 'calibParams-ballet.txt';
%calibinname = 'calibParams-ballet_vsrs.txt';
calibinname = 'cam_param_dancer.txt';
%calibinname = 'cam_param_balloons.txt';
zmin = 2289;
%zmin = 42;
%zmin = 448.251214;
zmax = 213500;
%zmax = 130;
%zmax = 11206.280350;
s = [1088 1920];
%s = [768 1024];


%% Read images

cams = read_calib_data(calibinname);

X = zeros(1,s(1)*s(2)*numel(views));
Y = zeros(1,s(1)*s(2)*numel(views));
Z = zeros(1,s(1)*s(2)*numel(views));
C = zeros(s(1)*s(2)*numel(views),3);

p = 1;
Pc = zeros(3,1); Pw = Pc;
for idx = 1:numel(views)
    i = views(idx);
    %[y,u,v] = yuvread([inname '-color_' num2str(i)],frame);
    %d = yuvread([inname '-depth420_' num2str(i)],frame);
    [y,u,v] = yuvread([inname '_c_' num2str(i) '_1920x1088'],frame);
    d = yuvread([inname '_d_' num2str(i) '_1920x1088'],frame);
    %[y,u,v] = yuvread([inname '_' num2str(i)],frame);
    %d = yuvread(['depth_' inname '_' num2str(i)],frame);
    
    % Get camera parameters
    R = cams{i}.E(1:3,1:3);
    T = cams{i}.E(1:3,4);
    fx = cams{i}.K(1,1);
    fy = cams{i}.K(2,2);
    cx = cams{i}.K(1,3);
    cy = cams{i}.K(2,3);
    
    % Unquantize depth
    z = 1 ./ ( (d.*((1/zmin) - (1/zmax))./255) + (1/zmax) );
    
    z = 2000*ones(size(d));
    %d = yuvread([inname '_d_2_1920x1088'],frame);
    %z = 1 ./ ( (d.*((1/zmin) - (1/zmax))./255) + (1/zmax) );
    
    
    % convert to rgb
    u2 = imresize (u,2);
    v2 = imresize (v,2);
    yuv(:,:,1) = y/255;
    yuv(:,:,2) = (u2-128)/255;
    yuv(:,:,3) = (v2-128)/255;
  
    ff = yuv(:,:,2);    
    ff(ff<-0.5) = -0.5;
    ff(ff>0.5) = 0.5;
    yuv(:,:,2) = ff;    

    ff = yuv(:,:,3);    
    ff(ff<-0.5) = -0.5;
    ff(ff>0.5) = 0.5;
    yuv(:,:,3) = ff;    

	rgb = yuv2rgb(yuv);
    rgb2 = zeros(size(rgb));
    
    cams{2}.E(:,4) = -1*cams{2}.E(:,4);

    % Project each point in the image
    for r = 1:size(rgb,1),
        for c = 1:size(rgb,2),

          %if z(r,c) < 10000,
          x = (c-1);
          y = (r-1);
          
          Pc(1) = (x - cx) * z(r,c) / fx;
          Pc(2) = (y - cy) * z(r,c) / fy;
          Pc(3) = z(r,c);
          
          % Convert point to world coordinates (CUIDADO CON ESTO QUE ES EL
          % PROBLEMA DE QUE R Y T ESTEN DEFINIDAS DE MUNDO A CAMARA O AL
          % REVES
          %Pw = R'*(Pc-T); % EC. SI R,T estan definidas de mundo a camara (no es el caso para undo dancer)
          Pw = R*Pc+T; % EC. si R,T definidas de camara a mundo (caso undo dancer)
          Pw(4) = 1;

          pix = cams{2}.K*cams{2}.E*Pw;
    
          rr = round(pix(2)/pix(3))+1;
          cc = round(pix(1)/pix(3))+1;
          if rr>0 && rr<=size(rgb2,1),
              if cc>0 && cc<=size(rgb2,2),
                  rgb2(rr,cc,:) = rgb(r,c,:);
              end;
          end;
          
          %X(p) = Pw(1);
          %Y(p) = Pw(2);
          %Z(p) = Pw(3);
          %C(p,:) = rgb(r,c,:);

          %p = p+1;
          %end
          
        end;
    end;
    
end;

%% Show results
%figure(1),imshow(rgb);
%figure(2),imshow(d/255);
pc = pointCloud([X;Y;Z]','Color',C);
%figure(1),pcshow(pc); xlabel('X'); ylabel('Y'); zlabel('Z');

pcwrite(pc,'pc_dancer_2.ply','PLYFormat','binary');


