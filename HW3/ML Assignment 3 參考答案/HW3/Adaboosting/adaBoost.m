%%
% File Name: AdaBoost
% This is the implementation of the ada boost algorithm.
% Parameters - very easy to gues by name...
% Return values: i - hypothesis-index  vector.
%                t - threshhols vector
%                beta - weighted beta.
%%
function boosted = adaBoost(train, train_label, cycles)
    disp('running adaBoost algorithm');
    d = size(train); % dimension of train_data
	distribution = ones(1, d(1)) / d(1); % init. data's weights
	error = zeros(1, cycles); % train error(number of model worst answer)
	beta = zeros(1, cycles); %
	label = (train_label(:) >= 5); % contain the correct label per vector
	for j = 1 : cycles
        
        if (mod(j, 10) == 0)
            disp([j, cycles]);
        end
        
        
        [i, t] = weakLearner(distribution, train,label); 
        
        %fprintf('The feature in %d iteration is %d \n', j, i);
        %fprintf('The threshold in %d iteration is %d \n', j, t);
        %fprintf('The direction in every iteration: 1 is positive, 0 is negative\n\n');
        
        % calculate the error of weakest [feature, threshold]
        error(j) = distribution * abs(label - (train(:,i) >= t));
        
        % update model's weight using formula ln(beta(j))
        beta(j) = error(j) / (1 - error(j)); 
        fprintf('The updated weight of %d iter decision stump is %d\n\n', j, log(beta(j)));
        
        % function will return the diamond_t, [feature, threshold]
        boosted(j, :) = [beta(j), i, t]; 
        
        % update data's weight using formula:G(x) = g_t * a_t * x
        distribution = distribution .* exp(log(beta(j)) * (1 - abs(label - (train(:, i) >= t))))'; % ' is transpose
        distribution = distribution / sum(distribution);
        
    end
    