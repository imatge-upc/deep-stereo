function cam = read_calib_data(name)

data = importdata(name);
cams = data(1:24:end);
for idx = 1:numel(cams)
    c = cams(idx);
    cam{c+1}.id = c;
    cam{c+1}.K = reshape(data(24*c+2:24*c+10),3,3)';
    cam{c+1}.dist = reshape(data(24*c+11:24*c+12),1,2);
    cam{c+1}.E = reshape(data(24*c+13:24*c+24),4,3)';
end;
