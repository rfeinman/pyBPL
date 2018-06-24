save_dir = 'lib_data';
lib = loadlib;
fields = fieldnames(lib);

% Make directories
mkdir(save_dir);

% Process basic fields
for i=1:numel(fields)
  name = fields{i};
  if strcmp(name, 'endstate')
      % if 'endstate' field, pass
      continue;
  end
  value = lib.(name);
  if isstruct(value)
    save_dir1 = strcat(save_dir, '/', name);
    mkdir(save_dir1);
    fields1 = fieldnames(value);
    for j=1:numel(fields1)
        name1 = fields1{j};
        value1 = value.(name1);
        save_file(save_dir1, name1, value1);
    end
  elseif isa(value, 'SpatialModel')
      continue;
  else
      save_file(save_dir, name, value);
  end
end

% Process SpatialModel
save_dir1 = strcat(save_dir, '/Spatial');
mkdir(save_dir1);
list_SH = lib.Spatial.list_SH;
for i=1:numel(list_SH)
    save_dir2 = strcat(save_dir1, '/H', int2str(i-1));
    mkdir(save_dir2);
    H = list_SH{i};
    fields_H = fieldnames(H);
    for j=1:numel(fields_H)
        name = fields_H{j};
        if strcmp(name, 'endstate')
            continue;
        end
        value = H.(name);
        save_file(save_dir2, name, value);
    end
end

function save_file(base_folder, fname, value)
    fname = strcat(base_folder, '/', fname, '.mat');
    save(fname, 'value');
end