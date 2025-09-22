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
