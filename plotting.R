library(tidyverse)
rep_num = 50

res = read_csv("regret_no_kernel.csv")
res = res %>% pivot_longer(c("SPO_correct", "ETO_correct", "SPO_wrong", "ETO_wrong"), names_to = "methods", values_to = "regret")
res_summary = res %>% group_by(n, methods) %>% summarise(avg_regret = mean(regret), std = sd(regret))

res2 = read_csv("regret_kernel_SGD.csv")
colnames(res2)[1] = "SPO_kernel_SGD"
res2 = res2 %>% pivot_longer(c("SPO_kernel_SGD", "ETO_kernel"), names_to = "methods", values_to = "regret")
res_summary2 = res2 %>% group_by(n, methods) %>% summarise(avg_regret = mean(regret), std = sd(regret))

res3 = read_csv("regret_kernel_gurobi.csv")
colnames(res3)[1] = "SPO_kernel_gurobi"
res3 = res3 %>% pivot_longer(c("SPO_kernel_gurobi"), names_to = "methods", values_to = "regret")
res_summary3 = res3 %>% group_by(n, methods) %>% summarise(avg_regret = mean(regret), std = sd(regret))

res_summary = rbind(res_summary, res_summary2, res_summary3)
table(res_summary$methods)
res_summary$method_type = NA
res_summary$method_type[grepl("SPO_correct", res_summary$methods, fixed=T)] = "SPO+ (Gurobi)"
res_summary$method_type[grepl("SPO_wrong", res_summary$methods, fixed=T)] = "SPO+ (Gurobi)"
res_summary$method_type[grepl("gurobi", res_summary$methods, fixed=T)] = "SPO+ (Gurobi)"
res_summary$method_type[grepl("SGD", res_summary$methods, fixed=T)] = "SPO+ (SGD)"
res_summary$method_type[grepl("ETO", res_summary$methods, fixed=T)] = "ETO"
res_summary$method_type = factor(res_summary$method_type, levels = c("ETO","SPO+ (Gurobi)","SPO+ (SGD)"))
res_summary$setting = NA
res_summary$setting[grepl("correct", res_summary$methods)] = "Correct linear"
res_summary$setting[grepl("wrong", res_summary$methods)] = "Wrong linear"
res_summary$setting[grepl("kernel", res_summary$methods)] = "Kernel"
res_summary$setting = factor(res_summary$setting, levels = c("Correct linear",
                                                             "Wrong linear",
                                                             "Kernel"))
res_summary = res_summary %>% mutate(lb = avg_regret - 1.96*std/sqrt(rep_num), ub = avg_regret + 1.96*std/sqrt(rep_num))

regretplot = res_summary %>%
  mutate(avg_regret = avg_regret/res$zstar_avg_test[1],
         lb = lb/res$zstar_avg_test[1],
         ub = ub/res$zstar_avg_test[1]) %>%
  mutate(method = method_type, hypothesis = setting) %>%
  ggplot(aes(x = n, y = avg_regret, shape = hypothesis,
             color = method, fill = method, linetype=method)) +
  scale_shape_manual("Hypothesis", values=c(1,0,6)) +
  scale_linetype_discrete("Method") +
  scale_color_manual("Method", values = c("#F8766D", "#619CFF", "#00C1AA")) +
  scale_fill_manual("Method", values = c("#F8766D", "#619CFF", "#00C1AA")) +
  geom_line(size = 0.75) + geom_point(size = 2.5) +
  geom_ribbon(aes(ymin = lb, ymax = ub), alpha=0.3, color=NA) + scale_y_continuous(labels=scales::percent) +
  ylab("Relative Regret") + guides(color = guide_legend(override.aes = list(shape = NA))) + #+ guides(color = guide_legend(order = 1), shape=guide_legend("Hypothesis", order = 2), fill = "none") +
  theme_minimal(base_size = 10)
regretplot
ggsave('regret.pdf', plot=regretplot, dpi = 300, height = 4, width = 5)

