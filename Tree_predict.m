function y = Tree_predict(Trees,X,nu)
    y = Trees{1};
    for i = 2: length(Trees)
        y = y + nu*predict(Trees{i}, X);
    end
end