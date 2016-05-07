function [selected, time] =Alpha_Investing(X, y, wealth, delta_alpha)
% ALPHA_INVESTING Streamwise Feature Selection 
% 
% [f, time] = Alpha_Investing(X, y, wealth, delta_alpha)
%
% This code implements main streamwise feature selection (SFS), which is
% known as Alpha investing 
%
% INPUT 
%  X: features: Number of Observations by Features 
%  y: class labels 
%  wealth: 
%  delta_alpha: 
%
% OUTPUT
%  selected: feature selected by alpha investing 
%  time: evaluation time

if nargin < 3
  wealth = .5;
end
if nargin < 4
  delta_alpha = .5;
end

start = tic;

% n observations; p features
[n,p] = size(X);
% initially add constant term into the model
model = [1, zeros(1,p-1)];
error = Prediction_Error(X(:,model==1), y, Linear_Regression(X(:,model==1), y));

for i = 2:p
  alpha = wealth/(2*i);

  model(i) = 1;
  error_new = Prediction_Error(X(:,model==1), y, Linear_Regression(X(:,model==1), y));
  sigma2 = error/n;
  p_value = exp((error_new-error)/(2*sigma2));

  if p_value < alpha %feature i is accepted
    model(i) = 1;
    error = error_new;
    wealth = wealth + delta_alpha - alpha;
  else %feature i is discarded
    model(i) = 0;
    wealth = wealth - alpha;
  end
end
 
% train final model
w = zeros(p,1);
w(model==1,1) = Linear_Regression(X(:,model==1), y);
  
time = toc(start);
selected = find(model);

% Linear_Regression 
function w = Linear_Regression(X, y)
% LINEAR_REGRESSION Build a linear model 
%
% w = Linear_Regression(X, y)
% 
% This is not the most efficient way to find w!

% w = inv(X'*X)*X'*y;
w = (X'*X)\X'*y;


% Prediction_Error 
function error = Prediction_Error(X, y, w)
% PREDICTION_ERROR Measure sum squared error a linear model 
%
% error = Prediction_Error(X, y, w)
yhat = X*w;
error = sum((y-yhat).^2);
