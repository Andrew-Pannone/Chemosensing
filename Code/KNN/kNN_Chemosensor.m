%% Load Dataset

% Dataset -

% The goal is to load the entire dataset and run knn based on a random
% split in the ratio of 70/30 training/testing

clear
clc

dataset_name = 'Dataset name';

% Load specified dataset with number of classes and specified sensors
FOM_cell = load('Dataset name.mat');
FOM_cell = FOM_cell.AnalyteFOM;
Num_classes = 6;
Sensor_list = [2:7];

% Define Hyperparameters
Samples_per_class = 175;
train_split_size = 120;
Neighbors = 30;

% Specify FOMs being evaluated
FOM_select=1:20;

%% Concatenate data

for class = [2:1+Num_classes]
    for FOM = 1:length(FOM_select)

        temp_fom_mat = [];
        for device = Sensor_list % all devices are trained on

            if device == 2
                for iters = 2:51
                    temp_fom_mat = [temp_fom_mat; FOM_cell{class,device}{iters,1}{2,FOM_select(FOM)}];
                end
            else
                for iters = 2:26
                    temp_fom_mat = [temp_fom_mat; FOM_cell{class,device}{iters,1}{2,FOM_select(FOM)}];
                end
            end
        end
        FOM_mat(:,FOM,class-1) = temp_fom_mat;

    end
end
clear device Sensor_list FOM FOM_cell iters class temp_fom_mat

%% Put FOM Mat into simple class format for KNN

for FOM = 1:length(FOM_select)
    temp_fom_mat = [];
    for class = 1:Num_classes

        temp_fom_mat = [temp_fom_mat; FOM_mat(:,FOM,class)];

    end
    FOM_mat_KNN(:,FOM) = temp_fom_mat;
end

clear FOM class temp_fom_mat
%% Set Train and Test Indices

train_ind=[];
test_ind=[];

% Set Train Indices
for i=1:Num_classes
    train_ind=[train_ind Samples_per_class*(i-1)+randperm(Samples_per_class,train_split_size)];
end

% Set Test Indices
for i=1:length(FOM_mat_KNN)
    if(length(find(train_ind==i))==0)
        test_ind=[test_ind i];
    end
end

% Assign data according to train/test split
FOM_mat_KNN_train = FOM_mat_KNN(train_ind,:);
FOM_mat_KNN_test = FOM_mat_KNN(test_ind,:);

%% Normalization (z-transform)

for i=1:20

    % Normalize train set
    ztmean=mean(FOM_mat_KNN_train(:,i));
    ztstd=std(FOM_mat_KNN_train(:,i));
    FOM_mat_KNN_train(:,i)=(FOM_mat_KNN_train(:,i)-ztmean)/ztstd;

    % Normalize test set
    ztmean=mean(FOM_mat_KNN_test(:,i));
    ztstd=std(FOM_mat_KNN_test(:,i));
    FOM_mat_KNN_test(:,i)=(FOM_mat_KNN_test(:,i)-ztmean)/ztstd;

end

%% Generate Class labels

samples_per_train_class = train_split_size;
samples_per_test_class = Samples_per_class-train_split_size;

ONS = ones(samples_per_train_class,1);
tmp = [];
for i = 1:Num_classes
    tmp = [tmp;i*ONS];
end

data_store2(:,1) = tmp;
ONS = ones(samples_per_test_class,1);

%% Distance calculation

clear data_store

% Create a matrix that computes distance for each point
for FOM = 1:length(FOM_select)
    for point = 1:samples_per_test_class*Num_classes % Point of interest
        for dist = 1:samples_per_train_class*Num_classes % distance measurements

            dist_store(dist,point,FOM) = ((FOM_mat_KNN_test(point,FOM)-FOM_mat_KNN_train(dist,FOM))); %L1 distance

        end
    end
end

%% k-NN loop

tic
counter=1;
for dimensions = 20%1:20

    % List of FOMs to evaluate
    Feature_list = nchoosek(1:20,dimensions);

    % ID of FOM list
    FOM_List = [1:length(Feature_list)];

    % Avoiding nchoosek function error for 20 inputs
    if dimensions == 20
        FOM_List = 1;
    end

    for FOM_Eval = FOM_List

        % create data storage variable for specified FOM
        data_store = [];
        for dim = 1:dimensions
            data_store(:,dim) = FOM_mat_KNN_test(:,Feature_list(FOM_Eval,dim));
        end

        % Produce class labels
        tmp = [];
        for i = 1:Num_classes
            tmp = [tmp;i*ONS];
        end

        data_store(:,21) = tmp;

        % Do distance calculations
        KNN_dist_temp = sum(abs(dist_store(:,:,Feature_list(FOM_Eval,:))),3); %L1

        % Do KNN Calculations
        temp_dist = zeros(samples_per_test_class*Num_classes,1);

        NN = Neighbors;
        NN_Class_store = zeros(length(FOM_mat_KNN_test),1);

        for point = 1:samples_per_test_class*Num_classes % Point of interest

            temp_dist = KNN_dist_temp(:,point);
            % Sort into ascending order
            [sort_list sort_order] = sort(temp_dist);

            % Take votes class of closest N neighbors
            clear votes
            votes = data_store2(sort_order(1:NN),1);

            % election
            votes_per_class = zeros(Num_classes,1);
            for i = 1:Num_classes
                votes_per_class(i,1) = sum(votes==i);
            end

            class_vote = find(votes_per_class==max(votes_per_class));

            % Tiebreaker
            while length(class_vote)>1

                NN = NN-1;
                votes = data_store2(sort_order(1:NN),1);

                % election
                votes_per_class = zeros(Num_classes,1);
                for i = 1:Num_classes
                    votes_per_class(i,1) = sum(votes==i);
                end
                class_vote = find(votes_per_class==max(votes_per_class));
            end

            NN_Class_store(point,1) = class_vote;

        end

        % Check Accuracy
        total_correct = zeros(1);
        test_size = samples_per_test_class*Num_classes;

        for i = 1:Num_classes  % Iterate the true label

            % Take every guess corresponding to true label i
            class_list_test = NN_Class_store(find(data_store(:,21)==i),1);

            % Number of predictions for a specific class
            num_predictions_test = length(find(class_list_test==i));
            total_correct = total_correct+num_predictions_test;

        end
        output(FOM_Eval,dimensions) = 100*total_correct/test_size;
        
        % Print Progress
        if rem(counter,100) == 0
            fprintf('%s dim %i $%f\n', dataset_name, dimensions,(100*counter)/1048575)
        end
        counter=counter+1;
    end

end
toc

save(['train_ind_', dataset_name, '.mat'],"train_ind")
save(['test_ind_', dataset_name, '.mat'],"test_ind")
save(['output_', dataset_name, '.mat'],"output")
