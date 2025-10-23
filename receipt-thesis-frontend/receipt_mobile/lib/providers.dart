import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'api_client.dart';
import 'models.dart';

final apiProvider = Provider<ApiClient>((ref) => ApiClient());

final healthProvider = FutureProvider<Map<String, dynamic>>((ref) {
  return ref.read(apiProvider).health();
});

final receiptsProvider = FutureProvider.autoDispose<List<Receipt>>((ref) {
  return ref.read(apiProvider).listReceipts(limit: 100, offset: 0);
});

final summaryProvider = FutureProvider<SummaryStats>((ref) {
  return ref.read(apiProvider).getSummary();
});

final byCategoryProvider = FutureProvider<List<Map<String, dynamic>>>((ref) {
  return ref.read(apiProvider).statsByCategory();
});

final topMerchantsProvider = FutureProvider<List<MerchantStat>>((ref) {
  return ref.read(apiProvider).topMerchants(limit: 5);
});

final weekdaySpendProvider = FutureProvider<List<WeekdayStat>>((ref) {
  return ref.read(apiProvider).weekdaySpend();
});

final rolling30Provider = FutureProvider<List<RollingStat>>((ref) {
  return ref.read(apiProvider).rolling30DaySpend();
});

final lowConfidenceProvider = FutureProvider<List<Receipt>>((ref) {
  return ref.read(apiProvider).lowConfidenceReceipts(threshold: 0.6, limit: 50);
});
