class Receipt {
  final String id;
  final String? store;
  final String? storeNormalized;
  final String? date; // ISO YYYY-MM-DD
  final double? total;
  final String? category;
  final String? categorySource;
  final double? confidence;
  final double? ocrConf;
  final String? createdAt;

  Receipt({
    required this.id,
    this.store,
    this.storeNormalized,
    this.date,
    this.total,
    this.category,
    this.categorySource,
    this.confidence,
    this.ocrConf,
    this.createdAt,
  });

  factory Receipt.fromJson(Map<String, dynamic> j) => Receipt(
        id: j['id'] ?? '',
        store: j['store'],
        storeNormalized: j['store_normalized'],
        date: j['date'],
        total: (j['total'] == null) ? null : (j['total'] as num).toDouble(),
        category: j['category'],
        categorySource: j['category_source'],
        confidence: (j['confidence'] == null) ? null : (j['confidence'] as num).toDouble(),
        ocrConf: (j['ocr_conf'] == null) ? null : (j['ocr_conf'] as num).toDouble(),
        createdAt: j['created_at'],
      );
}

class UploadResult {
  final String id;
  final String? store;
  final String? storeNormalized;
  final String? date;
  final double? total;
  final String? category;
  final double? confidence;
  final String? categorySource;
  final String text;        // raw OCR text
  final double? ocrConf;
  final bool yoloUsed;
  final bool ocrSpaceUsed;
  final bool ocrSpaceOk;
  final String? ocrSource;  // tesseract | ocr_space | consensus
  final String? ocrSourceLabel;
  final String? reason;

  UploadResult({
    required this.id,
    required this.text,
    this.store,
    this.storeNormalized,
    this.date,
    this.total,
    this.category,
    this.confidence,
    this.categorySource,
    this.ocrConf,
    this.yoloUsed = false,
    this.ocrSpaceUsed = false,
    this.ocrSpaceOk = false,
    this.ocrSource,
    this.ocrSourceLabel,
    this.reason,
  });

  factory UploadResult.fromJson(Map<String, dynamic> j) => UploadResult(
        id: j['id'] ?? '',
        store: j['store'],
        storeNormalized: j['store_normalized'],
        date: j['date'],
        total: (j['total'] == null) ? null : (j['total'] as num).toDouble(),
        category: j['category'],
        confidence: (j['confidence'] == null) ? null : (j['confidence'] as num).toDouble(),
        categorySource: j['category_source'],
        text: j['text'] ?? '',
        ocrConf: (j['ocr_conf'] == null) ? null : (j['ocr_conf'] as num).toDouble(),
        yoloUsed: j['yolo_used'] == true,
        ocrSpaceUsed: j['ocr_space_used'] == true,
        ocrSpaceOk: j['ocr_space_ok'] == true,
        ocrSource: j['ocr_source'],
        ocrSourceLabel: j['ocr_source_label'],
        reason: j['reason'],
      );
}

class ReceiptJobStatus {
  final String jobId;
  final String status;
  final String? filename;
  final DateTime? createdAt;
  final DateTime? startedAt;
  final DateTime? finishedAt;
  final UploadResult? result;
  final String? error;

  const ReceiptJobStatus({
    required this.jobId,
    required this.status,
    this.filename,
    this.createdAt,
    this.startedAt,
    this.finishedAt,
    this.result,
    this.error,
  });

  bool get isFinal => status == 'completed' || status == 'failed';

  factory ReceiptJobStatus.fromJson(Map<String, dynamic> j) {
    UploadResult? result;
    final rawResult = j['result'];
    if (rawResult is Map<String, dynamic>) {
      result = UploadResult.fromJson(rawResult);
    }

    DateTime? parseSeconds(dynamic value) {
      if (value is num) {
        final millis = (value * 1000).toInt();
        return DateTime.fromMillisecondsSinceEpoch(millis, isUtc: true).toLocal();
      }
      return null;
    }

    final rawId = j['id'] ?? j['job_id'];
    final jobId = rawId is String ? rawId : rawId?.toString() ?? '';

    return ReceiptJobStatus(
      jobId: jobId,
      status: (j['status'] ?? 'queued') as String,
      filename: j['filename'] as String?,
      createdAt: parseSeconds(j['created_at']),
      startedAt: parseSeconds(j['started_at']),
      finishedAt: parseSeconds(j['finished_at']),
      result: result,
      error: j['error'] as String?,
    );
  }
}

class SummaryStats {
  final double totalSpend;
  final int totalReceipts;
  final double monthToDateSpend;
  final String? topCategory;
  final double topCategoryTotal;

  SummaryStats({
    required this.totalSpend,
    required this.totalReceipts,
    required this.monthToDateSpend,
    required this.topCategory,
    required this.topCategoryTotal,
  });

  factory SummaryStats.fromJson(Map<String, dynamic> j) => SummaryStats(
        totalSpend: (j['total_spend'] as num?)?.toDouble() ?? 0.0,
        totalReceipts: j['total_receipts'] ?? 0,
        monthToDateSpend: (j['month_to_date_spend'] as num?)?.toDouble() ?? 0.0,
        topCategory: j['top_category'],
        topCategoryTotal: (j['top_category_total'] as num?)?.toDouble() ?? 0.0,
      );
}

class MerchantStat {
  final String store;
  final int receiptCount;
  final double totalSpend;

  MerchantStat({
    required this.store,
    required this.receiptCount,
    required this.totalSpend,
  });

  factory MerchantStat.fromJson(Map<String, dynamic> j) => MerchantStat(
        store: (j['store'] ?? 'Unknown') as String,
        receiptCount: j['receipt_count'] is int
            ? j['receipt_count'] as int
            : (j['receipt_count'] as num?)?.toInt() ?? 0,
        totalSpend: (j['total_spend'] as num?)?.toDouble() ?? 0.0,
      );
}

class WeekdayStat {
  final int weekday; // 0 = Sunday (Postgres) / SQLite, align with DateTime.weekday-1
  final double totalSpend;
  final int receiptCount;

  WeekdayStat({
    required this.weekday,
    required this.totalSpend,
    required this.receiptCount,
  });

  factory WeekdayStat.fromJson(Map<String, dynamic> j) => WeekdayStat(
        weekday: j['weekday'] is int ? j['weekday'] as int : (j['weekday'] as num?)?.toInt() ?? 0,
        totalSpend: (j['total_spend'] as num?)?.toDouble() ?? 0.0,
        receiptCount: j['receipt_count'] is int
            ? j['receipt_count'] as int
            : (j['receipt_count'] as num?)?.toInt() ?? 0,
      );
}

class RollingStat {
  final DateTime date;
  final double totalSpend;
  final int receiptCount;

  RollingStat({
    required this.date,
    required this.totalSpend,
    required this.receiptCount,
  });

  factory RollingStat.fromJson(Map<String, dynamic> j) {
    final raw = j['date'] as String?;
    final parsed = raw != null ? DateTime.tryParse(raw) : null;
    return RollingStat(
      date: parsed ?? DateTime.now(),
      totalSpend: (j['total_spend'] as num?)?.toDouble() ?? 0.0,
      receiptCount: j['receipt_count'] is int
          ? j['receipt_count'] as int
          : (j['receipt_count'] as num?)?.toInt() ?? 0,
    );
  }
}

/// User-defined custom category label
class CustomLabel {
  final String id;
  final String name;
  final String? color;
  final String? icon;
  final String? description;
  final int usageCount;
  final String? createdAt;
  final String? updatedAt;

  CustomLabel({
    required this.id,
    required this.name,
    this.color,
    this.icon,
    this.description,
    this.usageCount = 0,
    this.createdAt,
    this.updatedAt,
  });

  factory CustomLabel.fromJson(Map<String, dynamic> j) => CustomLabel(
        id: j['id'] ?? '',
        name: j['name'] ?? '',
        color: j['color'],
        icon: j['icon'],
        description: j['description'],
        usageCount: (j['usage_count'] as num?)?.toInt() ?? 0,
        createdAt: j['created_at'],
        updatedAt: j['updated_at'],
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'color': color,
        'icon': icon,
        'description': description,
        'usage_count': usageCount,
      };
}

/// Response from /categories endpoint
class CategoriesResponse {
  final List<CategoryItem> builtin;
  final List<CategoryItem> custom;

  CategoriesResponse({required this.builtin, required this.custom});

  factory CategoriesResponse.fromJson(Map<String, dynamic> j) {
    final builtinList = (j['builtin'] as List?)
            ?.map((e) => CategoryItem.fromJson(e as Map<String, dynamic>))
            .toList() ??
        [];
    final customList = (j['custom'] as List?)
            ?.map((e) => CategoryItem.fromJson(e as Map<String, dynamic>))
            .toList() ??
        [];
    return CategoriesResponse(builtin: builtinList, custom: customList);
  }

  /// All categories (builtin + custom) as a flat list
  List<CategoryItem> get all => [...builtin, ...custom];
}

class CategoryItem {
  final String name;
  final String type; // 'builtin' or 'custom'
  final String? color;
  final String? icon;
  final String? id; // only for custom
  final int usageCount;

  CategoryItem({
    required this.name,
    required this.type,
    this.color,
    this.icon,
    this.id,
    this.usageCount = 0,
  });

  bool get isCustom => type == 'custom';

  factory CategoryItem.fromJson(Map<String, dynamic> j) => CategoryItem(
        name: j['name'] ?? '',
        type: j['type'] ?? 'builtin',
        color: j['color'],
        icon: j['icon'],
        id: j['id'],
        usageCount: (j['usage_count'] as num?)?.toInt() ?? 0,
      );
}
