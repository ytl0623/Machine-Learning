%%
% File Name: AdaBoost
% This is the implementation of the ada boost algorithm.
% Parameters - very easy to gues by name...
% Return values: i - hypothesis-index  vector.
%                t - threshhols vector
%                beta - weighted beta.
%%
function boosted=adaBoost(train,train_label,cycles)
    disp('running adaBoost algorithm');
    d=size(train);  %[100,256]
	distribution=ones(1,d(1))/d(1);  %1/200=0.005 size=1*200
	error=zeros(1,cycles);  %size=1*100
	beta=zeros(1,cycles);  %size=1*100
	label=(train_label(:)>=5);% contain the correct label per vector

	for j=1:cycles
        if(mod(j,10)==0)
            disp([j,cycles]);
        end
    [i,t]=weakLearner(distribution,train,label);
    error(j)=distribution*abs(label-(train(:,i)>=t));
    beta(j)=error(j)/(1-error(j));
    boosted(j,:)=[beta(j),i,t];
    
    distribution=distribution.* exp(log(beta(j))*(1-abs(label-(train(:,i)>=t))))';  %更新權重
    distribution=distribution/sum(distribution);
    end