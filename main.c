#include "backprop.h"
#include "layer.h"
#include "neuron.h"


layer *lay = NULL;
int n_layers;
int *n_neurons;
float a=0.15;
float **input;
float *cost;
float total_cost;
float **outputs;
int n_train;
int n=1;

int main(void)
{
    int i;
    srand(time(0));
    printf("Entrez le nombre de couches souhaite :\n");
    scanf("%d",&n_layers);
    n_neurons = (int*) malloc(n_layers * sizeof(int));
    memset(n_neurons,0,n_layers *sizeof(int));

    // lire le nombres de neurones par couche
    for(i=0;i<n_layers;i++)
    {
        printf("Entrez le nombre de neurones par couche[%d]: \n",i+1);
        scanf("%d",&n_neurons[i]);
    }

    printf("\n");

    // erreur en initialisation
    if(init()!= SUCCESS_INIT)
    {
        printf("Erreur en initialisation ...\n");
        exit(0);
    }


    printf("Entrez le nombre d'exemples d'apprentissage: \n");
    scanf("%d",&n_train);
    printf("\n");

    input = (float**) malloc(n_train * sizeof(float*));
    for(i=0;i<n_train;i++)
    {
        input[i] = (float*)malloc(n_neurons[0] * sizeof(float));
    }

    outputs = (float**) malloc(n_train* sizeof(float*));
    for(i=0;i<n_train;i++)
    {
        outputs[i] = (float*)malloc(n_neurons[n_layers-1] * sizeof(float));
    }

    cost = (float *) malloc(n_neurons[n_layers-1] * sizeof(float));
    memset(cost,0,n_neurons[n_layers-1]*sizeof(float));

    // appel à la fonction get_inputs
    get_inputs();
    // appel à la fonction get_outputs
    get_outputs();
    train_neural_net();
    test_nn();

    if(dinit()!= SUCCESS_DINIT)
    {
        printf("Erreur ...\n");
    }

    return 0;
}


int init()
{
    if(create_architecture() != SUCCESS_CREATE_ARCHITECTURE)
    {
        printf("erreur en création de l'architecture...\n");
        return ERR_INIT;
    }

    printf("Le reseau de neurone a ete bien cree...\n\n");
    return SUCCESS_INIT;
}

//Get Inputs
void  get_inputs()
{
    int i,j;

        for(i=0;i<n_train;i++)
        {
            printf("Donnez les entrées pour l'exemple d'apprentissage[%d]:\n",i);

            for(j=0;j<n_neurons[0];j++)
            {
                scanf("%f",&input[i][j]);
                
            }
            printf("\n");
        }
}

//Get ouputs
void get_outputs()
{
    int i,j;
    
    for(i=0;i<n_train;i++)
    {
        for(j=0;j<n_neurons[n_layers-1];j++)
        {
            printf("Donnez la sortie desiree pour l'exemple d'apprentissage[%d]: \n",i);
            scanf("%f",&outputs[i][j]);
            printf("\n");
        }
    }
}

// ajouter les inputs à la couche d'entrée
void add_input(int i)
{
    int j;

    for(j=0;j<n_neurons[0];j++)
    {
        lay[0].neu[j].actv = input[i][j];
        printf("Input: %f\n",lay[0].neu[j].actv);
    }
}

// Crée l'architecture du réseau de neurone
int create_architecture()
{
    int i=0,j=0;
    lay = (layer*) malloc(n_layers * sizeof(layer));

    for(i=0;i<n_layers;i++)
    {
        lay[i] = create_layer(n_neurons[i]);      
        lay[i].num_neu = n_neurons[i];
        printf("La couche %d a ete cree\n", i+1);
        printf("le nombre des neurones dans la couche %d: %d\n", i+1,lay[i].num_neu);

        for(j=0;j<n_neurons[i];j++)
        {
            if(i < (n_layers-1)) 
            {
                lay[i].neu[j] = create_neuron(n_neurons[i+1]);
            }

            printf("le neurone %d dans la couche %d a été crée\n",j+1,i+1);  
        }
        printf("\n");
    }

    printf("\n");

    // Initialiser les poids
    if(initialize_weights() != SUCCESS_INIT_WEIGHTS)
    {
        printf("erreur...\n");
        return ERR_CREATE_ARCHITECTURE;
    }

    return SUCCESS_CREATE_ARCHITECTURE;
}

int initialize_weights(void)
{
    int i,j,k;

    if(lay == NULL)
    {
        printf("erreur...\n");
        return ERR_INIT_WEIGHTS;
    }

    printf("Initialisation des poids...\n");

    for(i=0;i<n_layers-1;i++)
    {
        
        for(j=0;j<n_neurons[i];j++)
        {
            for(k=0;k<n_neurons[i+1];k++)
            {
                // Initialisation des poids des sorties pour chaque neuron
                lay[i].neu[j].out_weights[k] = ((double)rand())/((double)RAND_MAX);
                printf("%d:w[%d][%d]: %f\n",k,i,j, lay[i].neu[j].out_weights[k]);
                lay[i].neu[j].dw[k] = 0.0;
            }

            if(i>0) 
            {
                lay[i].neu[j].bias = ((double)rand())/((double)RAND_MAX);
            }
        }
    }   
    printf("\n");
    
    for (j=0; j<n_neurons[n_layers-1]; j++)
    {
        lay[n_layers-1].neu[j].bias = ((double)rand())/((double)RAND_MAX);
    }

    return SUCCESS_INIT_WEIGHTS;
}

// l'apprentissage du réseau de neurones
void train_neural_net(void)
{
    int i;
    int it=0;

    // Gradient Descent
    for(it=0;it<20000;it++)
    {
        for(i=0;i<n_train;i++)
        {
            add_input(i);
            forward_prop();
            calculer_loss(i);
            back_prop(i);
            set_poids();
        }
    }
}



void set_poids(void)
{
    int i,j,k;

    for(i=0;i<n_layers-1;i++)
    {
        for(j=0;j<n_neurons[i];j++)
        {
            for(k=0;k<n_neurons[i+1];k++)
            {
                // mettre à jour les poids
                lay[i].neu[j].out_weights[k] = (lay[i].neu[j].out_weights[k]) - (a * lay[i].neu[j].dw[k]);
            }
            
            // mettre à jour le biais
            lay[i].neu[j].bias = lay[i].neu[j].bias - (a * lay[i].neu[j].dbias);
        }
    }   
}

void forward_prop(void)
{
    int i,j,k;

    for(i=1;i<n_layers;i++)
    {   
        for(j=0;j<n_neurons[i];j++)
        {
            lay[i].neu[j].z = lay[i].neu[j].bias;

            for(k=0;k<n_neurons[i-1];k++)
            {
                lay[i].neu[j].z  = lay[i].neu[j].z + ((lay[i-1].neu[k].out_weights[j])* (lay[i-1].neu[k].actv));
            }

            // la fonction d'activation ReLu pour les couches cachées
            if(i < n_layers-1)
            {
                if((lay[i].neu[j].z) < 0)
                {
                    lay[i].neu[j].actv = 0;
                }

                else
                {
                    lay[i].neu[j].actv = lay[i].neu[j].z;
                }
            }
            
            // la fonction d'activation sigmoid pour la couche de sortie
            else
            {
                lay[i].neu[j].actv = 1/(1+exp(-lay[i].neu[j].z));
                printf("Output: %d\n", (int)round(lay[i].neu[j].actv));
                printf("\n");
            }
        }
    }
}

// la loss functionn
void calculer_loss(int i)
{
    int j;
    float tmpcost=0;
    float tcost=0;

    for(j=0;j<n_neurons[n_layers-1];j++)
    {
        tmpcost = outputs[i][j] - lay[n_layers-1].neu[j].actv;
        cost[j] = (tmpcost * tmpcost)/2;
        tcost = tcost + cost[j];
    }   

    total_cost = (total_cost + tcost)/n;
    n++;
}

// l'erreur de la backprop
void back_prop(int p)
{
    int i,j,k;

    // la couche de retour
    for(j=0;j<n_neurons[n_layers-1];j++)
    {           
        lay[n_layers-1].neu[j].dz = (lay[n_layers-1].neu[j].actv - outputs[p][j]) * (lay[n_layers-1].neu[j].actv) * (1- lay[n_layers-1].neu[j].actv);

        for(k=0;k<n_neurons[n_layers-2];k++)
        {   
            lay[n_layers-2].neu[k].dw[j] = (lay[n_layers-1].neu[j].dz * lay[n_layers-2].neu[k].actv);
            lay[n_layers-2].neu[k].dactv = lay[n_layers-2].neu[k].out_weights[j] * lay[n_layers-1].neu[j].dz;
        }
            
        lay[n_layers-1].neu[j].dbias = lay[n_layers-1].neu[j].dz;           
    }

    // les couches cachées
    for(i=n_layers-2;i>0;i--)
    {
        for(j=0;j<n_neurons[i];j++)
        {
            if(lay[i].neu[j].z >= 0)
            {
                lay[i].neu[j].dz = lay[i].neu[j].dactv;
            }
            else
            {
                lay[i].neu[j].dz = 0;
            }

            for(k=0;k<n_neurons[i-1];k++)
            {
                lay[i-1].neu[k].dw[j] = lay[i].neu[j].dz * lay[i-1].neu[k].actv;    
                
                if(i>1)
                {
                    lay[i-1].neu[k].dactv = lay[i-1].neu[k].out_weights[j] * lay[i].neu[j].dz;
                }
            }

            lay[i].neu[j].dbias = lay[i].neu[j].dz;
        }
    }
}

// Tester notre RNN après l'apprentissage
void test_nn(void) 
{
    int i;
    while(1)
    {
        printf("Donnez les entrees à tester:\n");

        for(i=0;i<n_neurons[0];i++)
        {
            scanf("%f",&lay[0].neu[i].actv);
        }
        forward_prop();
    }
}



int dinit(void)
{
    return SUCCESS_DINIT;
}