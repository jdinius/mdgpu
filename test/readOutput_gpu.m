function out = readOutput_gpu(file,N)

fid = fopen(file,'r');
count = 1;
while ~feof(fid)
    try
        time(count) = fread(fid,1,'single');
        for i = 1:N
            pos(2*i-1:2*i,count) = fread(fid,2,'single');
            vel(2*i-1:2*i,count) = fread(fid,2,'single');
        end
        lyapExp(:,count) = fread(fid,4*N,'single');
        coll(count) = fread(fid,1,'int');
        count = count + 1;
    catch
        break
    end
end
fclose(fid);
figure;
plot(lyapExp(:,end))

out = lyapExp(:,end);