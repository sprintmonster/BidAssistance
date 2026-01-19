import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import type { Page } from "../../types/navigation";
import {
  FileText,
  Image as ImageIcon,
  Link as LinkIcon,
  Megaphone,
  Paperclip,
  Plus,
  Search,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

type NoticeCategory = "전체" | "서비스" | "업데이트" | "점검" | "정책";

type AttachmentKind = "image" | "pdf" | "file" | "link";

export interface NoticeAttachment {
  id: string;
  kind: AttachmentKind;
  name: string;
  url: string; // DataURL 또는 외부 URL
  mimeType?: string;
  size?: number;
}

export interface NoticeItem {
  id: number;
  title: string;
  category: Exclude<NoticeCategory, "전체">;
  date: string; // YYYY-MM-DD
  author: string;
  pinned?: boolean;
  content: string;
  attachments?: NoticeAttachment[];
}

const STORAGE_KEY = "notices.v1";

/**
 * 프로토타입 기본 데이터(초기 시드).
 * 실제 운영 시에는 서버에서 내려받는 형태로 교체 권장.
 */
const DEFAULT_NOTICES: NoticeItem[] = [
  {
    id: 1,
    title: "서비스 점검 안내 (01/10 02:00~04:00)",
    category: "점검",
    date: "2026-01-10",
    author: "관리자",
    pinned: true,
    content:
      "안정적인 서비스 제공을 위해 시스템 점검이 진행됩니다.\n점검 시간 동안 일부 기능이 제한될 수 있습니다.\n\n- 점검 일시: 2026-01-10 02:00 ~ 04:00\n- 영향 범위: 로그인, 알림 일부 지연 가능\n\n이용에 불편을 드려 죄송합니다.",
    attachments: [
      {
        id: "seed-link-1",
        kind: "link",
        name: "점검 상세 안내(외부 링크)",
        url: "https://example.com/maintenance",
      },
    ],
  },
  {
    id: 2,
    title: "입찰 알림 필터 기능 업데이트",
    category: "업데이트",
    date: "2026-01-08",
    author: "Product Team",
    content:
      "알림 필터가 개선되었습니다.\n\n- 알림 유형별(마감/정정/재공고 등) 필터\n- 관심 지역 기반 우선순위\n- 읽지 않음만 보기\n\n설정은 [알림] 페이지에서 확인할 수 있습니다.",
  },
  {
    id: 3,
    title: "개인정보 처리방침 변경 안내",
    category: "정책",
    date: "2026-01-05",
    author: "법무/정책",
    content:
      "개인정보 처리방침이 일부 개정되었습니다.\n\n- 수집 항목 문구 정비\n- 보관 및 파기 절차 보완\n\n자세한 내용은 [개인정보처리방침]을 참고해 주세요.",
  },
];

interface NoticePageProps {
  onNavigate: (page: Page, bidId?: number) => void;

  /**
   * 외부 주입(서버 연동 시 사용).
   * 미지정 시 localStorage(없으면 DEFAULT_NOTICES)로 동작.
   */
  notices?: NoticeItem[];

  /**
   * 작성/삭제 권한(운영 시 관리자만 true 권장)
   * 기본값 true는 데모 편의용.
   */
  canWrite?: boolean;
}

function safeParseNotices(raw: string | null): NoticeItem[] | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return null;
    return parsed as NoticeItem[];
  } catch {
    return null;
  }
}

function formatLocalDateYYYYMMDD(d: Date) {
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function makeId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function kindFromFile(file: File): AttachmentKind {
  if (file.type.startsWith("image/")) return "image";
  if (file.type === "application/pdf") return "pdf";
  return "file";
}

function fileToDataURL(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("파일을 읽는 중 오류가 발생했습니다."));
    reader.onload = () => resolve(String(reader.result));
    reader.readAsDataURL(file);
  });
}

/**
 * localStorage 저장은 용량 제한이 있으므로, 데모 안전장치로 제한을 둡니다.
 * (브라우저별로 다르지만 보통 5~10MB 수준)
 */
const MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024; // 2MB
const MAX_TOTAL_ATTACH_BYTES = 6 * 1024 * 1024; // 6MB

export function NoticePage({ onNavigate, notices: externalNotices, canWrite = true }: NoticePageProps) {
  const [notices, setNotices] = useState<NoticeItem[]>(() => {
    if (externalNotices) return externalNotices;
    const saved = safeParseNotices(localStorage.getItem(STORAGE_KEY));
    return saved && saved.length > 0 ? saved : DEFAULT_NOTICES;
  });

  useEffect(() => {
    if (externalNotices) return; // 서버 주입 모드면 localStorage 저장하지 않음
    localStorage.setItem(STORAGE_KEY, JSON.stringify(notices));
  }, [notices, externalNotices]);

  // 목록/검색
  const [category, setCategory] = useState<NoticeCategory>("전체");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);

  // 작성 폼
  const [composeOpen, setComposeOpen] = useState(false);
  const [formTitle, setFormTitle] = useState("");
  const [formAuthor, setFormAuthor] = useState("관리자");
  const [formDate, setFormDate] = useState(formatLocalDateYYYYMMDD(new Date()));
  const [formCategory, setFormCategory] = useState<Exclude<NoticeCategory, "전체">>("서비스");
  const [formPinned, setFormPinned] = useState(false);
  const [formContent, setFormContent] = useState("");
  const [formAttachments, setFormAttachments] = useState<NoticeAttachment[]>([]);
  const [linkName, setLinkName] = useState("");
  const [linkUrl, setLinkUrl] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const base = notices
      .slice()
      .sort((a, b) => {
        const pa = a.pinned ? 1 : 0;
        const pb = b.pinned ? 1 : 0;
        if (pa !== pb) return pb - pa;
        return b.date.localeCompare(a.date);
      });

    return base.filter((n) => {
      const catOk = category === "전체" ? true : n.category === category;
      const qOk =
        !q ||
        n.title.toLowerCase().includes(q) ||
        n.content.toLowerCase().includes(q) ||
        n.author.toLowerCase().includes(q);
      return catOk && qOk;
    });
  }, [notices, category, query]);

  const categories: NoticeCategory[] = ["전체", "서비스", "업데이트", "점검", "정책"];

  const selected = useMemo(() => {
    if (selectedId == null) return null;
    return notices.find((n) => n.id === selectedId) ?? null;
  }, [selectedId, notices]);

  const totalAttachBytes = useMemo(() => {
    return formAttachments.reduce((sum, a) => sum + (a.size ?? 0), 0);
  }, [formAttachments]);

  const resetCompose = () => {
    setFormTitle("");
    setFormAuthor("관리자");
    setFormDate(formatLocalDateYYYYMMDD(new Date()));
    setFormCategory("서비스");
    setFormPinned(false);
    setFormContent("");
    setFormAttachments([]);
    setLinkName("");
    setLinkUrl("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const openCompose = () => {
    resetCompose();
    setComposeOpen(true);
  };

  const addLinkAttachment = () => {
    const name = linkName.trim();
    const url = linkUrl.trim();
    if (!name || !url) {
      toast.info("링크 이름과 URL을 입력해 주세요.");
      return;
    }
    if (!/^https?:\/\//i.test(url)) {
      toast.error("링크 URL은 http:// 또는 https://로 시작해야 합니다.");
      return;
    }
    setFormAttachments((prev) => [
      ...prev,
      { id: makeId(), kind: "link", name, url },
    ]);
    setLinkName("");
    setLinkUrl("");
  };

  const removeAttachment = (id: string) => {
    setFormAttachments((prev) => prev.filter((a) => a.id !== id));
  };

  const handleFilesSelected = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const next: NoticeAttachment[] = [];
    let addedBytes = 0;

    for (const f of Array.from(files)) {
      if (f.size > MAX_FILE_SIZE_BYTES) {
        toast.error(`파일이 너무 큽니다: ${f.name} (최대 2MB)`);
        continue;
      }
      if (totalAttachBytes + addedBytes + f.size > MAX_TOTAL_ATTACH_BYTES) {
        toast.error("첨부파일 총 용량이 너무 큽니다. (최대 6MB)");
        break;
      }

      try {
        const url = await fileToDataURL(f);
        next.push({
          id: makeId(),
          kind: kindFromFile(f),
          name: f.name,
          url,
          mimeType: f.type,
          size: f.size,
        });
        addedBytes += f.size;
      } catch {
        toast.error(`파일을 처리할 수 없습니다: ${f.name}`);
      }
    }

    if (next.length > 0) setFormAttachments((prev) => [...prev, ...next]);
  };

  const createNotice = () => {
    const title = formTitle.trim();
    const author = formAuthor.trim();
    const date = formDate.trim();
    const content = formContent.trim();

    if (!title) {
      toast.error("제목을 입력해 주세요.");
      return;
    }
    if (!author) {
      toast.error("작성자를 입력해 주세요.");
      return;
    }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      toast.error("날짜 형식이 올바르지 않습니다. (YYYY-MM-DD)");
      return;
    }
    if (!content) {
      toast.error("내용을 입력해 주세요.");
      return;
    }

    const nextId = notices.reduce((max, n) => Math.max(max, n.id), 0) + 1;

    const newItem: NoticeItem = {
      id: nextId,
      title,
      author,
      date,
      category: formCategory,
      pinned: formPinned,
      content,
      attachments: formAttachments.length > 0 ? formAttachments : undefined,
    };

    setNotices((prev) => [newItem, ...prev]);
    setComposeOpen(false);
    toast.success("공지사항이 등록되었습니다.");
  };

  const deleteNotice = (id: number) => {
    const ok = window.confirm("이 공지사항을 삭제할까요?");
    if (!ok) return;
    setNotices((prev) => prev.filter((n) => n.id !== id));
    setSelectedId((prev) => (prev === id ? null : prev));
    toast.success("삭제되었습니다.");
  };

  const AttachmentIcon = ({ kind }: { kind: AttachmentKind }) => {
    if (kind === "image") return <ImageIcon className="h-4 w-4" />;
    if (kind === "pdf") return <FileText className="h-4 w-4" />;
    if (kind === "link") return <LinkIcon className="h-4 w-4" />;
    return <Paperclip className="h-4 w-4" />;
  };

  const renderAttachment = (a: NoticeAttachment) => {
    if (a.kind === "image") {
      return (
        <div key={a.id} className="space-y-2">
          <div className="text-sm font-medium flex items-center gap-2">
            <AttachmentIcon kind={a.kind} />
            <span className="truncate">{a.name}</span>
          </div>
          <a href={a.url} target="_blank" rel="noreferrer">
            <img
              src={a.url}
              alt={a.name}
              className="w-full max-h-64 object-contain rounded-md border bg-white"
            />
          </a>
          <a className="text-sm underline text-blue-600" href={a.url} download={a.name}>
            다운로드
          </a>
        </div>
      );
    }

    if (a.kind === "pdf") {
      return (
        <PdfAttachment key={a.id} name={a.name} url={a.url} />
      );
    }

    // file / link
    return (
      <div key={a.id} className="flex items-center justify-between gap-3 border rounded-md p-3 bg-white">
        <div className="flex items-center gap-2 min-w-0">
          <AttachmentIcon kind={a.kind} />
          <div className="min-w-0">
            <div className="text-sm font-medium truncate">{a.name}</div>
            <div className="text-xs text-muted-foreground truncate">{a.url}</div>
          </div>
        </div>
        <div className="shrink-0">
          <a
            className="text-sm underline text-blue-600"
            href={a.url}
            target={a.kind === "link" ? "_blank" : undefined}
            rel={a.kind === "link" ? "noreferrer" : undefined}
            download={a.kind === "file" ? a.name : undefined}
          >
            {a.kind === "link" ? "열기" : "다운로드"}
          </a>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Megaphone className="h-6 w-6" />
            공지사항
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            서비스 업데이트, 점검, 정책 변경 소식을 확인하세요.
          </p>
        </div>

        <div className="flex gap-2">
          <Button variant="outline" onClick={() => onNavigate("home")}>
            홈
          </Button>
          {canWrite && (
            <Button className="gap-2" onClick={openCompose}>
              <Plus className="h-4 w-4" />
              공지 작성
            </Button>
          )}
        </div>
      </div>

      {/* 작성 폼 */}
      {composeOpen && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">공지 작성</CardTitle>
            <CardDescription>
              제목/내용/첨부파일을 등록합니다. (데모 모드: localStorage 저장)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label>제목</Label>
                <Input value={formTitle} onChange={(e) => setFormTitle(e.target.value)} placeholder="제목 입력" />
              </div>
              <div className="space-y-2">
                <Label>작성자</Label>
                <Input value={formAuthor} onChange={(e) => setFormAuthor(e.target.value)} placeholder="예: 관리자" />
              </div>
              <div className="space-y-2">
                <Label>날짜 (YYYY-MM-DD)</Label>
                <Input value={formDate} onChange={(e) => setFormDate(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>카테고리</Label>
                <div className="flex flex-wrap gap-2">
                  {(["서비스", "업데이트", "점검", "정책"] as const).map((c) => (
                    <Button
                      key={c}
                      size="sm"
                      variant={formCategory === c ? "default" : "outline"}
                      onClick={() => setFormCategory(c)}
                    >
                      {c}
                    </Button>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                type="button"
                size="sm"
                variant={formPinned ? "default" : "outline"}
                onClick={() => setFormPinned((v) => !v)}
              >
                {formPinned ? "고정됨" : "고정하기"}
              </Button>
              <span className="text-xs text-muted-foreground">
                고정 공지는 목록 최상단에 노출됩니다.
              </span>
            </div>

            <div className="space-y-2">
              <Label>내용</Label>
              <textarea
                value={formContent}
                onChange={(e) => setFormContent(e.target.value)}
                placeholder="내용 입력"
                className="w-full min-h-40 rounded-md border px-3 py-2 text-sm bg-white"
              />
            </div>

            <div className="space-y-2">
              <Label>첨부파일 (이미지/PDF/일반파일)</Label>
              <Input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={(e) => void handleFilesSelected(e.target.files)}
              />
              <div className="text-xs text-muted-foreground">
                데모 모드(localStorage 저장): 파일 1개 최대 2MB, 총 6MB 제한
              </div>

              {formAttachments.length > 0 && (
                <div className="space-y-2">
                  <div className="text-sm font-medium">
                    첨부 목록 ({formAttachments.length}개)
                  </div>
                  <div className="space-y-2">
                    {formAttachments.map((a) => (
                      <div key={a.id} className="flex items-center justify-between gap-3 border rounded-md p-3 bg-white">
                        <div className="flex items-center gap-2 min-w-0">
                          <AttachmentIcon kind={a.kind} />
                          <div className="min-w-0">
                            <div className="text-sm font-medium truncate">{a.name}</div>
                            <div className="text-xs text-muted-foreground">
                              {a.kind.toUpperCase()}
                              {typeof a.size === "number" ? ` · ${(a.size / 1024).toFixed(0)}KB` : ""}
                            </div>
                          </div>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => removeAttachment(a.id)}>
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label>외부 링크 첨부(선택)</Label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <Input value={linkName} onChange={(e) => setLinkName(e.target.value)} placeholder="링크 이름" />
                <Input value={linkUrl} onChange={(e) => setLinkUrl(e.target.value)} placeholder="https://..." />
                <Button variant="outline" onClick={addLinkAttachment}>
                  링크 추가
                </Button>
              </div>
            </div>

            <div className="flex gap-2 justify-end pt-2">
              <Button variant="outline" onClick={() => setComposeOpen(false)}>
                취소
              </Button>
              <Button onClick={createNotice}>등록</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 상세 보기 */}
      {selected && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-2">
                <div className="flex items-center gap-2 flex-wrap">
                  {selected.pinned && <Badge>고정</Badge>}
                  <Badge variant="secondary">{selected.category}</Badge>
                  <span className="text-xs text-muted-foreground">{selected.date}</span>
                  <span className="text-xs text-muted-foreground">·</span>
                  <span className="text-xs text-muted-foreground">{selected.author}</span>
                  {selected.attachments && selected.attachments.length > 0 && (
                    <span className="text-xs text-muted-foreground">
                      · 첨부 {selected.attachments.length}개
                    </span>
                  )}
                </div>
                <CardTitle className="text-lg">{selected.title}</CardTitle>
              </div>

              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={() => setSelectedId(null)}>
                  목록으로
                </Button>
                {canWrite && (
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={() => deleteNotice(selected.id)}
                    className="gap-2"
                  >
                    <Trash2 className="h-4 w-4" />
                    삭제
                  </Button>
                )}
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-6">
            <div className="whitespace-pre-line text-sm text-gray-700 leading-6">
              {selected.content}
            </div>

            {selected.attachments && selected.attachments.length > 0 && (
              <div className="space-y-3">
                <div className="text-sm font-semibold flex items-center gap-2">
                  <Paperclip className="h-4 w-4" />
                  첨부파일
                </div>

                {/* 이미지 먼저 보여주기 */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {selected.attachments.map(renderAttachment)}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* 검색/필터 */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">검색 및 필터</CardTitle>
          <CardDescription>제목/내용/작성자로 검색할 수 있습니다.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="notice-search">검색</Label>
            <div className="relative">
              <Search className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <Input
                id="notice-search"
                placeholder="예: 점검, 업데이트, 정책, 관리자"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {categories.map((c) => (
              <Button
                key={c}
                size="sm"
                variant={category === c ? "default" : "outline"}
                onClick={() => setCategory(c)}
              >
                {c}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 목록 */}
      <div className="space-y-3">
        {filtered.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              검색 결과가 없습니다.
            </CardContent>
          </Card>
        )}

        {filtered.map((n) => {
          const attachCount = n.attachments?.length ?? 0;
          return (
            <Card key={n.id} className={n.pinned ? "border-blue-200" : ""}>
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      {n.pinned && <Badge>고정</Badge>}
                      <Badge variant="secondary">{n.category}</Badge>
                      <span className="text-xs text-muted-foreground">{n.date}</span>
                      <span className="text-xs text-muted-foreground">·</span>
                      <span className="text-xs text-muted-foreground">{n.author}</span>
                      {attachCount > 0 && (
                        <span className="text-xs text-muted-foreground">· 첨부 {attachCount}개</span>
                      )}
                    </div>

                    <CardTitle className="text-lg truncate">{n.title}</CardTitle>
                  </div>

                  <Button size="sm" variant="outline" onClick={() => setSelectedId(n.id)}>
                    상세
                  </Button>
                </div>
              </CardHeader>
            </Card>
          );
        })}
      </div>
    </div>
  );
}

function PdfAttachment({ name, url }: { name: string; url: string }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2 border rounded-md p-3 bg-white">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="h-4 w-4" />
          <div className="min-w-0">
            <div className="text-sm font-medium truncate">{name}</div>
            <div className="text-xs text-muted-foreground truncate">PDF</div>
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <Button size="sm" variant="outline" onClick={() => setOpen((v) => !v)}>
            {open ? "미리보기 닫기" : "미리보기"}
          </Button>
          <a className="text-sm underline text-blue-600 self-center" href={url} download={name}>
            다운로드
          </a>
        </div>
      </div>

      {open && (
        <div className="border rounded-md overflow-hidden bg-white">
          <iframe title={name} src={url} className="w-full h-[520px]" />
        </div>
      )}
    </div>
  );
}
