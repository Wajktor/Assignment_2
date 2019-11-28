DATA = load("-ascii", "data.txt");

GPU = DATA(1:2:end,:);
CPU = DATA(2:2:end,:);

legend_names = [];
for i = 1:25
  
  plot(GPU(:,2), GPU(:,3))
  hold on;
  legend_names = [legend_names; strcat("TPB = ", num2str( GPU(i,1 ) ) )]

end

legend(legend_names)
xlabel("Number of particles")
ylabel("Elapsed time [s]")