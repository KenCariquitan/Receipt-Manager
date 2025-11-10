import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'api_client.dart';
import 'models.dart';

final apiProvider = Provider<ApiClient>((ref) => ApiClient());

final sessionProvider = StreamProvider<Session?>((ref) async* {
  final auth = Supabase.instance.client.auth;
  yield auth.currentSession;
  yield* auth.onAuthStateChange.map((event) => event.session);
});

Session _requireSession(Ref ref) {
  ref.watch(sessionProvider);
  final session = Supabase.instance.client.auth.currentSession;
  if (session == null) {
    throw Exception('Not signed in');
  }
  return session;
}

final healthProvider = FutureProvider<Map<String, dynamic>>((ref) {
  return ref.read(apiProvider).health();
});

final receiptsProvider =
    FutureProvider.autoDispose<List<Receipt>>((ref) async {
  _requireSession(ref);
  return ref.read(apiProvider).listReceipts(limit: 100, offset: 0);
});

final summaryProvider =
    FutureProvider.autoDispose<SummaryStats>((ref) async {
  _requireSession(ref);
  return ref.read(apiProvider).getSummary();
});

final byCategoryProvider =
    FutureProvider.autoDispose<List<Map<String, dynamic>>>((ref) async {
  _requireSession(ref);
  return ref.read(apiProvider).statsByCategory();
});

final topMerchantsProvider =
    FutureProvider.autoDispose<List<MerchantStat>>((ref) async {
  _requireSession(ref);
  return ref.read(apiProvider).topMerchants(limit: 5);
});

final weekdaySpendProvider =
    FutureProvider.autoDispose<List<WeekdayStat>>((ref) async {
  _requireSession(ref);
  return ref.read(apiProvider).weekdaySpend();
});

final rolling30Provider =
    FutureProvider.autoDispose<List<RollingStat>>((ref) async {
  _requireSession(ref);
  return ref.read(apiProvider).rolling30DaySpend();
});

final lowConfidenceProvider =
    FutureProvider.autoDispose<List<Receipt>>((ref) async {
  _requireSession(ref);
  return ref.read(apiProvider).lowConfidenceReceipts(threshold: 0.6, limit: 50);
});
