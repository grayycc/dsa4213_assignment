# Part III error analysis

Best LM representation by validation macro-F1: **lstm_w2v:mean**

## Most common confusions (true -> predicted)

| true | predicted | count | share of errors |
|---|---|---|---|
| results | methods | 656 | 15.2% |
| background | conclusions | 475 | 11.0% |
| objective | background | 450 | 10.4% |
| methods | results | 436 | 10.1% |
| conclusions | results | 391 | 9.0% |
| results | conclusions | 330 | 7.6% |

## Accuracy by sentence position in the abstract

| representation | first | middle | last |
|---|---|---|---|
| lstm_w2v:mean | 0.623 | 0.804 | 0.763 |
| transformer:mean | 0.605 | 0.800 | 0.754 |
| lstm:mean | 0.575 | 0.763 | 0.712 |
| rnn:mean | 0.572 | 0.771 | 0.700 |
| tfidf | 0.604 | 0.827 | 0.809 |

## Misclassified examples (lstm_w2v:mean), from the top confusions

- true **results**, predicted **methods** (sentence 6/13): psychologically , melancholic patients were defined by introversion , and a greater use of suppression to regulate negative emotion .
- true **results**, predicted **methods** (sentence 13/15): results for twice-daily co-trimoxazole for @ days and once-daily co-trimoxazole for @ days were similar .
- true **background**, predicted **conclusions** (sentence 2/12): in the mental health field , interventions that consider the patient-centred perspective -- such as patient quality of life , involvement in the treatment , treatment satisfaction , and working alliance -- have increased and better clinical outcomes discovered for patients with schizophrenia .
- true **background**, predicted **conclusions** (sentence 2/8): abpm is a very effective technique for assessing biological , environmental , and drug effects on blood pressure .
- true **objective**, predicted **background** (sentence 3/12): this study was designed to evaluate the efficacy and complications of dilation-assisted stone extraction for cbd stones .
- true **objective**, predicted **background** (sentence 1/9): most research on the illness management and recovery ( imr ) program for people with severe mental illnesses has focused on individuals with stable symptoms living in the community , with less attention to persons being treated in an inpatient setting .
- true **methods**, predicted **results** (sentence 5/13): nurses caring for infants in study group @ used the moldable positioner .
- true **methods**, predicted **results** (sentence 2/13): positively aspherized-pals have near addition and high positive distance zone aspherization comparable to the addition power .

## transformer:mean right, lstm:mean wrong (1917 sentences); examples

- true **methods**, lstm:mean said **results** (sentence 8/19): @ h and @ h after disease onset , the cranial ct was re-examined .
- true **methods**, lstm:mean said **results** (sentence 7/16): vessel pattern , lesion size and iodine staining .
- true **results**, lstm:mean said **methods** (sentence 5/11): half of patients were in cid for > @ % of their followup time .
- true **background**, lstm:mean said **conclusions** (sentence 2/10): these elevated concentrations are associated with cardiovascular and all-cause mortality .

## lstm:mean right, transformer:mean wrong (1177 sentences); examples

- true **conclusions**, transformer:mean said **background** (sentence 13/13): pain in hysteroscopy is significantly related to the presence of cervical synechiae , to the duration of the procedure , and to the use of normal saline ; conversely , parity seems to have a protective role .
- true **background**, transformer:mean said **conclusions** (sentence 2/11): this also holds true for the exchange of health-related information in a variety of internet forums .
- true **methods**, transformer:mean said **results** (sentence 11/18): a high positive pressure followed by a precipitous pressure drop , without negative peak pressure upon entering the es .
- true **background**, transformer:mean said **objective** (sentence 2/19): we assessed the efficacy and safety of nivolumab compared with investigator 's choice of chemotherapy ( icc ) as a second-line or later-line treatment in patients with advanced melanoma .
