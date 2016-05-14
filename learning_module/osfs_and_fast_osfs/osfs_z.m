function   [selected_features, time]=osfs_z(data,class_index,alpha)
% for continue value

[n,p] = size(data);
ns = max(data);
selected_features = [];
selected_features1 = [];

start = tic;
for i = 1:p-1
  %for very sparse data 
  n1 = sum(data(:,i));
  if n1 == 0
    continue;
  end
  
  stop = 0;
  [CI, dep] = my_cond_indep_fisher_z(data, i, class_index, [], n, alpha);
  if CI == 1 || isnan(dep)
    continue;
  end
  
  if CI == 0
    stop = 1;
    selected_features = [selected_features, i];
  end
  
  if stop
    p2 = length(selected_features);
    selected_features1 = selected_features;
    for j = 1:p2
      b = setdiff(selected_features1, selected_features(j), 'stable');
      if ~isempty(b)
        [CI, dep] = compter_dep_2(b, selected_features(j), class_index, 3, 0, alpha, 'z', data);
        if CI == 1 || isnan(dep)
          selected_features1 = b;
        end
      end
    end
  end
  selected_features = selected_features1;
end

time = toc(start);
  
    
      