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
        reason: j['reason'],
      );
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
