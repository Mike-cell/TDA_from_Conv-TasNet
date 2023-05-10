mixture_train_100_mix_clean = '/home/nsbao/disk1/xbz/lzd/Sepformer/LibriMix/data/Libri2Mix/wav8k/min/metadata/mixture_train-100_mix_clean.csv'

# awk -F, '{print $1 ".wav " "/path/to/files/" $1 ".wav"}' metrics_dev_mix_clean.csv > output.txt
# cut -d ',' -f 1,2 data.csv > output.txt
cut -d ' ' -f 1,2 mixture_train-100_mix_clean.csv > tr_mix.scp
cut -d ' ' -f 1,3 mixture_train-100_mix_clean.csv > tr_s1.scp
cut -d ' ' -f 1,4 mixture_train-100_mix_clean.csv > tr_s2.scp
cut -d ',' -f 1,2 mixture_train-100_mix_clean.csv | sed 's/,/ /g' | awk '{print $1 " " $2}' ORS='\r\n' > tr_mix.scp
cut -d ',' -f 1,3 mixture_train-100_mix_clean.csv | sed 's/,/ /g' | awk '{print $1 " " $2}' ORS='\r\n' > tr_s1.scp
cut -d ',' -f 1,4 mixture_train-100_mix_clean.csv | sed 's/,/ /g' | awk '{print $1 " " $2}' ORS='\r\n' > tr_s2.scp

cut -d ',' -f 1,2 mixture_test_mix_clean.csv | sed 's/,/ /g' | awk '{print $1 " " $2}' ORS='\r\n' > tt_mix.scp
cut -d ',' -f 1,3 mixture_test_mix_clean.csv | sed 's/,/ /g' | awk '{print $1 " " $2}' ORS='\r\n' > tt_s1.scp
cut -d ',' -f 1,4 mixture_test_mix_clean.csv | sed 's/,/ /g' | awk '{print $1 " " $2}' ORS='\r\n' > tt_s2.scp

train_mix_scp = 'tr_mix.scp'
train_s1_scp = 'tr_s1.scp'
train_s2_scp = 'tr_s2.scp'

test_mix_scp = 'tt_mix.scp'
test_s1_scp = 'tt_s1.scp'
test_s2_scp = 'tt_s2.scp'

