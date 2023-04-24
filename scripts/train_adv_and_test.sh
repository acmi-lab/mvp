BATCH_SIZE=$1
DATASET=$2
MODEL=$3
MODEL_TYPE=$4
EPOCHS=$5
LR=${6}
POOL_LABELS=${7}
POOL_TEMPLATES=${8}
POOL_LABELS_TEST=${9}
POOL_TEMPLATES_TEST=${10}
TEMPLATE_FILE=${11}
VERBALIZER_FILE=${12}
EXTRA_NAMES=${13}
ATTACK=${14}
MODE=${15}
NUM_TEMPLATE=${16}  
TRAIN_SIZE=${17} 
VAL_SIZE=${18}
EPSILON=${19}
NORM=${20}
NUM_ITER=${21}

source ~/.bashrc
echo $PWD


for SEED in 0 1 2;
do
    echo $SEED
    MODEL_ID=${MODEL_TYPE}_${SEED}_${EXTRA_NAMES}
    MODELPATH=./checkpoints/${DATASET}/${MODEL}/model_${MODEL_ID}/

    mkdir -p ${MODELPATH}
    nohup python main.py  --mode $MODE \
                    --dataset $DATASET \
                    --model_type $MODEL_TYPE \
                    --model_id $MODEL_ID \
                    --batch_size $BATCH_SIZE \
                    --model $MODEL \
                    --num_epochs $EPOCHS \
                    --lr $LR  \
                    --pool_label_words $POOL_LABELS \
                    --pool_templates $POOL_TEMPLATES \
                    --verbalizer_file $VERBALIZER_FILE \
                    --template_file $TEMPLATE_FILE \
                    --num_template $NUM_TEMPLATE \
                    --train_size $TRAIN_SIZE \
                    --path None \
                    --seed $SEED \
                    --patience 10 --adv_augment 1 \
                    --epsilon $EPSILON \
                    --norm $NORM \
                    --num_iter $NUM_ITER \
                    --val_size $VAL_SIZE > ${MODELPATH}/logs_trainer.txt

    

    nohup nice -n10 python main.py --mode attack \
                                --path ${MODELPATH}/final_model/ \
                                --attack_name textfooler \
                                --num_examples 1000 --dataset ${DATASET} \
                                --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                                --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
                                --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
                                --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED > ${MODELPATH}/logs_textfooler.txt


    nohup nice -n10 python main.py --mode attack \
                            --path ${MODELPATH}/final_model/ \
                            --attack_name textbugger \
                            --num_examples 1000 --dataset ${DATASET} \
                            --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                            --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
                            --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE}  \
                            --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED > ${MODELPATH}/logs_textbugger.txt



done