By default, in the multivariate output case, apply_output_fun (a.k.a. h) returns an array of arrays. There are two options to handle this:

1. Make apply_output_fun use the vcat(vec_of_vecs...)-trick inside of apply_output_fun
2. Store the output of the h-function in a temp-vector, and then use the vcat(temp...)-trick on that vector

I think option 2 is better!! More transparent what is actually happening. I think both options work both in multivariate and scalar case, as the vcat(...)-trick has no effect in the case when the output is just a scalar.

However, testing all this is difficult without running SGD, and we can't run SGD since we have no sensitivity model for the reactor. SO YOU SHOULD CREATE A SENSITIVITY MODEL FOR THE REACTOR, AT LEAST FOR ONE PARAMETER!!!!