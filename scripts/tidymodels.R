library(librarian)
librarian::shelf(dplyr, tidymodels, ranger, kernlab, palmerpenguins, tidyr)

penguins_clean <- penguins %>% 
  drop_na() %>% 
  mutate(species = factor(species),
        sex = factor(sex),
        island = factor(island))

table(penguins_clean$sex)
table(penguins_clean$species)

set.seed(42)
peng_split <- rsample::initial_split(penguins_clean, 0.85, strata = species)
peng_train_df <- rsample::training(peng_split)
peng_test_df <- rsample::testing(peng_split)

peng_train_split <- initial_split(peng_train_df, prop = 70/85, strata = species)
peng_assess <- testing(peng_train_split)
peng_analysis <- training(peng_train_split)

blr_mdl <- parsnip::logistic_reg() %>% 
  set_engine('glm')

peng_fit1 <- blr_mdl %>% 
  parsnip::fit(sex ~ species + bill_length_mm + bill_depth_mm + 
  flipper_length_mm + body_mass_g, data = peng_analysis)

peng_fit2 <- blr_mdl %>% 
  fit(sex ~ species + bill_length_mm + bill_depth_mm, data = peng_analysis)

peng_fit3 <- blr_mdl %>% 
  fit(sex ~ species + island + year, data = peng_analysis)
# broom::tidy(peng_fit1)

peng_pred1 <- peng_assess %>% 
  mutate(predict(peng_fit1, new_data = peng_assess, type = 'class'))
peng_pred3 <- peng_assess %>% 
  mutate(predict(peng_fit3, new_data = peng_assess, type = 'class'))

peng_pred1 %>% select(sex, .pred_class) %>% table()
peng_pred3 %>% select(sex, .pred_class) %>% table()

peng_train_folds <- vfold_cv(peng_train_df, v = 5, repeats = 3)

blr_mdl <- logistic_reg() %>% 
  set_engine('glm')

blr_wf <- workflows::workflow() %>% 
  add_model(blr_mdl) %>% 
  add_formula(sex ~ species + bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g)

blr_fit_folds <- blr_wf %>% 
  tune::fit_resamples(resamples = peng_train_folds)

tune::collect_metrics(blr_fit_folds)

spec1 <- sex ~ species + bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g
spec2 <- sex ~ species + bill_length_mm + bill_depth_mm
spec3 <- sex ~ species + island + year

blr_mdl <- parsnip::logistic_reg() %>%
  parsnip::set_engine('glm')

rf_mdl <- parsnip::rand_forest(trees = 1000) %>%
  parsnip::set_engine('ranger') %>%
  parsnip::set_mode('classification')
  
svm_mdl <- parsnip::svm_linear() %>%
  parsnip::set_engine('kernlab') %>%
  parsnip::set_mode('classification')

blr_spec1 <- workflow() %>%
  add_model(blr_mdl) %>%
  add_formula(spec1) %>%
  fit_resamples(peng_train_folds)

blr_spec2 <- workflow() %>%
  add_model(blr_mdl) %>%
  add_formula(spec2) %>%
  fit_resamples(peng_train_folds)

blr_spec3 <- workflow() %>%
  add_model(blr_mdl) %>%
  add_formula(spec3) %>%
  fit_resamples(peng_train_folds)

collect_metrics(blr_spec1, type = 'wide')
collect_metrics(blr_spec2, type = 'wide')
collect_metrics(blr_spec3, type = 'wide')

spec_list <- list(spec1, spec2, spec3)

spec_list <- list(spec1, spec2, spec3)

for(spec in spec_list) {
  rf_specs <- workflow() %>%
    add_model(rf_mdl) %>%
    add_formula(spec) %>%
    fit_resamples(peng_train_folds)
  
  results <- collect_metrics(rf_specs, type = 'wide')
  print(results)
}

spec_list <- list(spec1, spec2, spec3)

svm_specs <- purrr::map(
  .x = spec_list,
  .f = function(spec) {
    mdl <- workflow() %>%
      add_model(svm_mdl) %>%
      add_formula(spec) %>%
      fit_resamples(peng_train_folds)
    
    collect_metrics(mdl, type = 'wide')
  })

svm_specs_df <- svm_specs %>%
  setNames(as.character(spec_list)) %>%
  bind_rows(.id = 'model_spec')

svm_specs_df

### set up our final model as a workflow
final_workflow <- workflow() %>%
  add_model(svm_mdl) %>%
  add_formula(spec1)    ### specification with all biometric variables

final_fit <- last_fit(final_workflow, peng_split)

collect_metrics(final_fit)


